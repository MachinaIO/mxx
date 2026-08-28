import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard114
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard115
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1489
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1492
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1535

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound227899
def owner : Owner := ⟨.program ⟨257⟩, ⟨57104⟩⟩
def transferEvent : Nat := 227899
def frameStart : Nat := 227811
def rule : BoundRule := .product (.predecessor 0 227897 .coefficient) (.predecessor 1 227898 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 227897 .coefficient)
      LeftAuthority227872.bound (LeftAuthority227872.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events890.exact227873RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority227872.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority227872.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 227898 .coefficient)
      LeftAuthority227895.bound (LeftAuthority227895.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events890.exact227896RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority227895.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority227895.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority227872.bound LeftAuthority227895.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority227872.bound, LeftAuthority227895.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority227872.actual selector witness) * (LeftAuthority227895.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound227899

namespace LeftBound227907
def owner : Owner := ⟨.program ⟨257⟩, ⟨57105⟩⟩
def transferEvent : Nat := 227907
def frameStart : Nat := 227811
def rule : BoundRule := .sum [.predecessor 0 227905 .coefficient, .predecessor 1 227906 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 227905 .coefficient)
      LeftAuthority227903.bound (LeftAuthority227903.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events890.exact227904RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority227903.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority227903.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 227906 .coefficient)
      LeftBound227899.bound (LeftBound227899.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events890.exact227901RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound227899.bound, RecordedBoundRefines] <;> decide)
      (LeftBound227899.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority227903.bound, LeftBound227899.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority227903.bound, LeftBound227899.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority227903.actual selector witness, LeftBound227899.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound227907

namespace LeftBound227911
def owner : Owner := ⟨.program ⟨257⟩, ⟨58886⟩⟩
def transferEvent : Nat := 227911
def frameStart : Nat := 227811
def rule : BoundRule := .sum [.predecessor 0 227909 .coefficient, .predecessor 1 227910 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 227909 .coefficient)
      LeftBound227907.bound (LeftBound227907.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events890.exact227908RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound227907.bound, RecordedBoundRefines] <;> decide)
      (LeftBound227907.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 227910 .coefficient)
      LeftBound227888.bound (LeftBound227888.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events890.exact227893RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound227888.bound, RecordedBoundRefines] <;> decide)
      (LeftBound227888.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound227907.bound, LeftBound227888.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound227907.bound, LeftBound227888.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound227907.actual selector witness, LeftBound227888.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound227911

namespace LeftBound227924
def owner : Owner := ⟨.program ⟨257⟩, ⟨58884⟩⟩
def transferEvent : Nat := 227924
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 227922 .coefficient, .predecessor 1 227923 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 227922 .coefficient)
      LeftBound227753.bound (LeftBound227753.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events890.exact227921RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound227753.bound, RecordedBoundRefines] <;> decide)
      (LeftBound227753.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 227923 .coefficient)
      LeftBound227736.bound (LeftBound227736.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events889.exact227743RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound227736.bound, RecordedBoundRefines] <;> decide)
      (LeftBound227736.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound227753.bound, LeftBound227736.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound227753.bound, LeftBound227736.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound227753.actual selector witness, LeftBound227736.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound227924

namespace LeftBound227927
def owner : Owner := ⟨.program ⟨257⟩, ⟨58884⟩⟩
def transferEvent : Nat := 227927
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 227921 .summary, .result 227743 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 227921 .summary)
      LeftBound227755.bound (LeftBound227755.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨57699⟩⟩) (rawTerms := some (Proof.Events890.exact227921RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound227755.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 227743 .summary)
      LeftBound227738.bound (LeftBound227738.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨58883⟩⟩) (rawTerms := some (Proof.Events889.exact227743RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound227738.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound227755.bound, LeftBound227738.bound]
def bound : CoeffClass := .finite ⟨32190182365603518530196853751808, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound227755.bound, LeftBound227738.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound227755.actual selector witness, LeftBound227738.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound227927

namespace LeftBound227951
def owner : Owner := ⟨.program ⟨257⟩, ⟨24759⟩⟩
def transferEvent : Nat := 227951
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 227949 .coefficient) (.predecessor 1 227950 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 227949 .coefficient)
      LeftAuthority10841.bound (LeftAuthority10841.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events042.exact10842RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority10841.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority10841.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 227950 .coefficient)
      LeftBound222151.bound (LeftBound222151.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events867.exact222153RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound222151.bound, RecordedBoundRefines] <;> decide)
      (LeftBound222151.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32768 ⟨true, false, none, none, none⟩ LeftAuthority10841.bound LeftBound222151.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority10841.bound, LeftBound222151.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := tensorFactor 32768 ⟨true, false, none, none, none⟩ * (LeftAuthority10841.actual selector witness) * (LeftBound222151.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound227951

namespace LeftBound227956
def owner : Owner := ⟨.program ⟨257⟩, ⟨8464⟩⟩
def transferEvent : Nat := 227956
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 227954 .coefficient) (.predecessor 1 227955 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 227954 .coefficient)
      LeftBound222022.bound (LeftBound222022.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events867.exact222023RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound222022.bound, RecordedBoundRefines] <;> decide)
      (LeftBound222022.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 227955 .coefficient)
      LeftBound23091.bound (LeftBound23091.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events090.exact23092RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound23091.bound, RecordedBoundRefines] <;> decide)
      (LeftBound23091.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound222022.bound LeftBound23091.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound222022.bound, LeftBound23091.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound222022.actual selector witness) * (LeftBound23091.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 40) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound227956

namespace LeftBound227961
def owner : Owner := ⟨.program ⟨257⟩, ⟨24760⟩⟩
def transferEvent : Nat := 227961
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 227959 .coefficient, .predecessor 1 227960 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 227959 .coefficient)
      LeftBound227956.bound (LeftBound227956.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events890.exact227958RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound227956.bound, RecordedBoundRefines] <;> decide)
      (LeftBound227956.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 227960 .coefficient)
      LeftBound227951.bound (LeftBound227951.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events890.exact227953RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound227951.bound, RecordedBoundRefines] <;> decide)
      (LeftBound227951.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound227956.bound, LeftBound227951.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound227956.bound, LeftBound227951.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound227956.actual selector witness, LeftBound227951.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound227961

namespace LeftBound227965
def owner : Owner := ⟨.program ⟨257⟩, ⟨24761⟩⟩
def transferEvent : Nat := 227965
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 227963 .coefficient, .predecessor 1 227964 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 227963 .coefficient)
      LeftBound227961.bound (LeftBound227961.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events890.exact227962RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound227961.bound, RecordedBoundRefines] <;> decide)
      (LeftBound227961.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 227964 .coefficient)
      LeftBound23083.bound (LeftBound23083.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events090.exact23084RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound23083.bound, RecordedBoundRefines] <;> decide)
      (LeftBound23083.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound227961.bound, LeftBound23083.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound227961.bound, LeftBound23083.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound227961.actual selector witness, LeftBound23083.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound227965

namespace LeftBound227966
def owner : Owner := ⟨.program ⟨257⟩, ⟨24761⟩⟩
def transferEvent : Nat := 227966
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨98⟩⟩]⟩ [⟨.result 23084 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 23084 .coefficient)
      LeftBound23083.bound (LeftBound23083.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨98⟩⟩) (rawTerms := some (Proof.Events090.exact23084RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound23083.bound, RecordedBoundRefines] <;> decide)
      (LeftBound23083.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound23083.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound23083.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftBound23083.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound227966

namespace LeftBound227971
def owner : Owner := ⟨.program ⟨257⟩, ⟨53501⟩⟩
def transferEvent : Nat := 227971
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 227969 .coefficient) (.predecessor 1 227970 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 227969 .coefficient)
      LeftBound227965.bound (LeftBound227965.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events890.exact227968RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound227965.bound, RecordedBoundRefines] <;> decide)
      (LeftBound227965.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 227970 .coefficient)
      LeftAuthority10844.bound (LeftAuthority10844.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events042.exact10845RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority10844.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority10844.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftBound227965.bound LeftAuthority10844.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound227965.bound, LeftAuthority10844.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftBound227965.actual selector witness) * (LeftAuthority10844.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound227971

namespace LeftBound227972
def owner : Owner := ⟨.program ⟨257⟩, ⟨53501⟩⟩
def transferEvent : Nat := 227972
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨257⟩, ⟨53498⟩⟩], []⟩ [⟨.result 10845 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 10845 .coefficient)
      LeftAuthority10844.bound (LeftAuthority10844.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨53498⟩⟩) (rawTerms := some (Proof.Events042.exact10845RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority10844.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority10844.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority10844.bound []
def bound : CoeffClass := .finite ⟨12, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority10844.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority10844.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound227972

namespace LeftBound227973
def owner : Owner := ⟨.program ⟨257⟩, ⟨53501⟩⟩
def transferEvent : Nat := 227973
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 227968 .summary) (.transfer 227972) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 227968 .summary)
      LeftBound227966.bound (LeftBound227966.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨24761⟩⟩) (rawTerms := some (Proof.Events890.exact227968RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound227966.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 227972)
      LeftBound227972.bound (LeftBound227972.actual selector witness) := by
  exact .transfer (LeftBound227972.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound227966.bound LeftBound227972.bound
def bound : CoeffClass := .finite ⟨10223616, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound227966.bound, LeftBound227972.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound227966.actual selector witness) * (LeftBound227972.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound227973

namespace LeftBound227979
def owner : Owner := ⟨.program ⟨257⟩, ⟨53502⟩⟩
def transferEvent : Nat := 227979
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 227977 .coefficient) (.predecessor 1 227978 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 227977 .coefficient)
      LeftAuthority10844.bound (LeftAuthority10844.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events042.exact10845RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority10844.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority10844.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 227978 .coefficient)
      LeftBound222151.bound (LeftBound222151.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events867.exact222153RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound222151.bound, RecordedBoundRefines] <;> decide)
      (LeftBound222151.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32768 ⟨true, false, none, none, none⟩ LeftAuthority10844.bound LeftBound222151.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority10844.bound, LeftBound222151.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := tensorFactor 32768 ⟨true, false, none, none, none⟩ * (LeftAuthority10844.actual selector witness) * (LeftBound222151.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound227979

namespace LeftBound227984
def owner : Owner := ⟨.program ⟨257⟩, ⟨8481⟩⟩
def transferEvent : Nat := 227984
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 227982 .coefficient) (.predecessor 1 227983 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 227982 .coefficient)
      LeftBound222022.bound (LeftBound222022.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events867.exact222023RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound222022.bound, RecordedBoundRefines] <;> decide)
      (LeftBound222022.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 227983 .coefficient)
      LeftBound23132.bound (LeftBound23132.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events090.exact23133RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound23132.bound, RecordedBoundRefines] <;> decide)
      (LeftBound23132.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound222022.bound LeftBound23132.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound222022.bound, LeftBound23132.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound222022.actual selector witness) * (LeftBound23132.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 40) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound227984

namespace LeftBound227989
def owner : Owner := ⟨.program ⟨257⟩, ⟨53503⟩⟩
def transferEvent : Nat := 227989
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 227987 .coefficient, .predecessor 1 227988 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 227987 .coefficient)
      LeftBound227984.bound (LeftBound227984.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events890.exact227986RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound227984.bound, RecordedBoundRefines] <;> decide)
      (LeftBound227984.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 227988 .coefficient)
      LeftBound227979.bound (LeftBound227979.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events890.exact227981RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound227979.bound, RecordedBoundRefines] <;> decide)
      (LeftBound227979.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound227984.bound, LeftBound227979.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound227984.bound, LeftBound227979.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound227984.actual selector witness, LeftBound227979.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound227989

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
