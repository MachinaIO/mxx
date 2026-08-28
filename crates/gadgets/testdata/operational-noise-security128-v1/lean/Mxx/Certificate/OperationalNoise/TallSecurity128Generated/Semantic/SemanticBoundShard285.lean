import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard276
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard283
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard284

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound47866
def owner : Owner := ⟨.program ⟨257⟩, ⟨42854⟩⟩
def transferEvent : Nat := 47866
def frameStart : Nat := 47764
def rule : BoundRule := .product (.predecessor 0 47864 .coefficient) (.predecessor 1 47865 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 47864 .coefficient)
      LeftAuthority47819.bound (LeftAuthority47819.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events186.exact47820RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority47819.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority47819.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 47865 .coefficient)
      LeftAuthority47862.bound (LeftAuthority47862.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events186.exact47863RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority47862.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority47862.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority47819.bound LeftAuthority47862.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority47819.bound, LeftAuthority47862.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority47819.actual selector witness) * (LeftAuthority47862.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound47866

namespace LeftBound47874
def owner : Owner := ⟨.program ⟨257⟩, ⟨42855⟩⟩
def transferEvent : Nat := 47874
def frameStart : Nat := 47764
def rule : BoundRule := .sum [.predecessor 0 47872 .coefficient, .predecessor 1 47873 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 47872 .coefficient)
      LeftAuthority47870.bound (LeftAuthority47870.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events186.exact47871RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority47870.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority47870.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 47873 .coefficient)
      LeftBound47866.bound (LeftBound47866.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events186.exact47868RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound47866.bound, RecordedBoundRefines] <;> decide)
      (LeftBound47866.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority47870.bound, LeftBound47866.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority47870.bound, LeftBound47866.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority47870.actual selector witness, LeftBound47866.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound47874

namespace LeftBound47878
def owner : Owner := ⟨.program ⟨257⟩, ⟨44391⟩⟩
def transferEvent : Nat := 47878
def frameStart : Nat := 47764
def rule : BoundRule := .sum [.predecessor 0 47876 .coefficient, .predecessor 1 47877 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 47876 .coefficient)
      LeftBound47874.bound (LeftBound47874.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events187.exact47875RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound47874.bound, RecordedBoundRefines] <;> decide)
      (LeftBound47874.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 47877 .coefficient)
      LeftBound47855.bound (LeftBound47855.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events186.exact47860RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound47855.bound, RecordedBoundRefines] <;> decide)
      (LeftBound47855.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound47874.bound, LeftBound47855.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound47874.bound, LeftBound47855.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound47874.actual selector witness, LeftBound47855.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound47878

namespace LeftBound47891
def owner : Owner := ⟨.program ⟨257⟩, ⟨44389⟩⟩
def transferEvent : Nat := 47891
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 47889 .coefficient, .predecessor 1 47890 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 47889 .coefficient)
      LeftBound47712.bound (LeftBound47712.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events187.exact47888RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound47712.bound, RecordedBoundRefines] <;> decide)
      (LeftBound47712.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 47890 .coefficient)
      LeftBound47695.bound (LeftBound47695.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events186.exact47702RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound47695.bound, RecordedBoundRefines] <;> decide)
      (LeftBound47695.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound47712.bound, LeftBound47695.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound47712.bound, LeftBound47695.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound47712.actual selector witness, LeftBound47695.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound47891

namespace LeftBound47894
def owner : Owner := ⟨.program ⟨257⟩, ⟨44389⟩⟩
def transferEvent : Nat := 47894
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 47888 .summary, .result 47702 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 47888 .summary)
      LeftBound47714.bound (LeftBound47714.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨43312⟩⟩) (rawTerms := some (Proof.Events187.exact47888RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound47714.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 47702 .summary)
      LeftBound47697.bound (LeftBound47697.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨44388⟩⟩) (rawTerms := some (Proof.Events186.exact47702RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound47697.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound47714.bound, LeftBound47697.bound]
def bound : CoeffClass := .finite ⟨2998273677530297008128, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound47714.bound, LeftBound47697.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound47714.actual selector witness, LeftBound47697.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound47894

namespace LeftBound47898
def owner : Owner := ⟨.program ⟨257⟩, ⟨44871⟩⟩
def transferEvent : Nat := 47898
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 47896 .coefficient) (.predecessor 1 47897 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 47896 .coefficient)
      LeftBound47891.bound (LeftBound47891.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events187.exact47895RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound47891.bound, RecordedBoundRefines] <;> decide)
      (LeftBound47891.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 47897 .coefficient)
      LeftAuthority47617.bound (LeftAuthority47617.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events186.exact47618RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority47617.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority47617.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound47891.bound LeftAuthority47617.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound47891.bound, LeftAuthority47617.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound47891.actual selector witness) * (LeftAuthority47617.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound47898

namespace LeftBound47899
def owner : Owner := ⟨.program ⟨257⟩, ⟨44871⟩⟩
def transferEvent : Nat := 47899
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨44869⟩⟩]⟩ [⟨.result 47618 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 47618 .coefficient)
      LeftAuthority47617.bound (LeftAuthority47617.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨44869⟩⟩) (rawTerms := some (Proof.Events186.exact47618RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority47617.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority47617.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority47617.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority47617.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority47617.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound47899

namespace LeftBound47900
def owner : Owner := ⟨.program ⟨257⟩, ⟨44871⟩⟩
def transferEvent : Nat := 47900
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 47895 .summary) (.transfer 47899) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 47895 .summary)
      LeftBound47894.bound (LeftBound47894.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨44389⟩⟩) (rawTerms := some (Proof.Events187.exact47895RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound47894.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 47899)
      LeftBound47899.bound (LeftBound47899.actual selector witness) := by
  exact .transfer (LeftBound47899.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound47894.bound LeftBound47899.bound
def bound : CoeffClass := .finite ⟨32193718473625689247691015454720, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound47894.bound, LeftBound47899.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound47894.actual selector witness) * (LeftBound47899.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound47900

namespace LeftBound47911
def owner : Owner := ⟨.program ⟨257⟩, ⟨43698⟩⟩
def transferEvent : Nat := 47911
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 47909 .coefficient) (.value (.predecessor 1 47910 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 47909 .coefficient)
      LeftAuthority47907.bound (LeftAuthority47907.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events187.exact47908RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority47907.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority47907.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 47910 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority47907.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority47907.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority47907.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound47911

namespace LeftBound47915
def owner : Owner := ⟨.program ⟨257⟩, ⟨43699⟩⟩
def transferEvent : Nat := 47915
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 47913 .coefficient) (.predecessor 1 47914 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 47913 .coefficient)
      LeftBound46742.bound (LeftBound46742.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events182.exact46745RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound46742.bound, RecordedBoundRefines] <;> decide)
      (LeftBound46742.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 47914 .coefficient)
      LeftBound47911.bound (LeftBound47911.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events187.exact47912RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound47911.bound, RecordedBoundRefines] <;> decide)
      (LeftBound47911.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound46742.bound LeftBound47911.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound46742.bound, LeftBound47911.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound46742.actual selector witness) * (LeftBound47911.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound47915

namespace LeftBound47916
def owner : Owner := ⟨.program ⟨257⟩, ⟨43699⟩⟩
def transferEvent : Nat := 47916
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨43696⟩⟩]⟩ [⟨.result 47908 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 47908 .coefficient)
      LeftAuthority47907.bound (LeftAuthority47907.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨43696⟩⟩) (rawTerms := some (Proof.Events187.exact47908RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority47907.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority47907.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority47907.bound []
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority47907.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority47907.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound47916

namespace LeftBound47917
def owner : Owner := ⟨.program ⟨257⟩, ⟨43699⟩⟩
def transferEvent : Nat := 47917
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 46745 .summary) (.transfer 47916) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 46745 .summary)
      LeftBound46743.bound (LeftBound46743.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨11216⟩⟩) (rawTerms := some (Proof.Events182.exact46745RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound46743.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 47916)
      LeftBound47916.bound (LeftBound47916.actual selector witness) := by
  exact .transfer (LeftBound47916.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound46743.bound LeftBound47916.bound
def bound : CoeffClass := .finite ⟨202072841853861888, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound46743.bound, LeftBound47916.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound46743.actual selector witness) * (LeftBound47916.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound47917

namespace LeftBound48012
def owner : Owner := ⟨.program ⟨257⟩, ⟨42853⟩⟩
def transferEvent : Nat := 48012
def frameStart : Nat := 47973
def rule : BoundRule := .identity (.predecessor 0 48011 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 48011 .coefficient)
      LeftAuthority48009.bound (LeftAuthority48009.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events187.exact48010RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority48009.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority48009.derived selector witness)

def rawBound : CoeffClass := LeftAuthority48009.bound
def bound : CoeffClass := .finite ⟨52, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority48009.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority48009.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound48012

namespace LeftBound48029
def owner : Owner := ⟨.program ⟨257⟩, ⟨44178⟩⟩
def transferEvent : Nat := 48029
def frameStart : Nat := 47973
def rule : BoundRule := .sum [.predecessor 0 48027 .coefficient, .predecessor 1 48028 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 48027 .coefficient)
      LeftBound48012.bound (LeftBound48012.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound48012.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 48028 .coefficient)
      LeftAuthority48025.bound (LeftAuthority48025.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority48025.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound48012.bound, LeftAuthority48025.bound]
def bound : CoeffClass := .finite ⟨52, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound48012.bound, LeftAuthority48025.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound48012.actual selector witness, LeftAuthority48025.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound48029

namespace LeftBound48032
def owner : Owner := ⟨.program ⟨257⟩, ⟨44179⟩⟩
def transferEvent : Nat := 48032
def frameStart : Nat := 47973
def rule : BoundRule := .identity (.predecessor 0 48031 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 48031 .coefficient)
      LeftBound48029.bound (LeftBound48029.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound48029.derived selector witness)

def rawBound : CoeffClass := LeftBound48029.bound
def bound : CoeffClass := .finite ⟨52, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound48029.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound48029.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound48032

namespace LeftBound48038
def owner : Owner := ⟨.program ⟨257⟩, ⟨44180⟩⟩
def transferEvent : Nat := 48038
def frameStart : Nat := 47973
def rule : BoundRule := .product (.predecessor 0 48036 .coefficient) (.predecessor 1 48037 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 48036 .coefficient)
      LeftAuthority48034.bound (LeftAuthority48034.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events187.exact48035RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority48034.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority48034.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 48037 .coefficient)
      LeftBound48032.bound (LeftBound48032.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events187.exact48033RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound48032.bound, RecordedBoundRefines] <;> decide)
      (LeftBound48032.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority48034.bound LeftBound48032.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority48034.bound, LeftBound48032.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority48034.actual selector witness) * (LeftBound48032.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound48038

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
