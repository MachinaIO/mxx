import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard107
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard678
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard681
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard682
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard717

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound110009
def owner : Owner := ⟨.program ⟨257⟩, ⟨59515⟩⟩
def transferEvent : Nat := 110009
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 110004 .summary) (.transfer 110008) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 110004 .summary)
      LeftBound110002.bound (LeftBound110002.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨25265⟩⟩) (rawTerms := some (Proof.Events429.exact110004RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound110002.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 110008)
      LeftBound110008.bound (LeftBound110008.actual selector witness) := by
  exact .transfer (LeftBound110008.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound110002.bound LeftBound110008.bound
def bound : CoeffClass := .finite ⟨15335424, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound110002.bound, LeftBound110008.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound110002.actual selector witness) * (LeftBound110008.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound110009

namespace LeftBound110015
def owner : Owner := ⟨.program ⟨257⟩, ⟨59516⟩⟩
def transferEvent : Nat := 110015
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 110013 .coefficient) (.predecessor 1 110014 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 110013 .coefficient)
      LeftAuthority4814.bound (LeftAuthority4814.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events018.exact4815RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4814.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority4814.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 110014 .coefficient)
      LeftBound105151.bound (LeftBound105151.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events410.exact105153RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound105151.bound, RecordedBoundRefines] <;> decide)
      (LeftBound105151.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32768 ⟨true, false, none, none, none⟩ LeftAuthority4814.bound LeftBound105151.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority4814.bound, LeftBound105151.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := tensorFactor 32768 ⟨true, false, none, none, none⟩ * (LeftAuthority4814.actual selector witness) * (LeftBound105151.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound110015

namespace LeftBound110020
def owner : Owner := ⟨.program ⟨257⟩, ⟨8711⟩⟩
def transferEvent : Nat := 110020
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 110018 .coefficient) (.predecessor 1 110019 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 110018 .coefficient)
      LeftBound105022.bound (LeftBound105022.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events410.exact105023RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound105022.bound, RecordedBoundRefines] <;> decide)
      (LeftBound105022.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 110019 .coefficient)
      LeftBound22130.bound (LeftBound22130.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events086.exact22131RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound22130.bound, RecordedBoundRefines] <;> decide)
      (LeftBound22130.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound105022.bound LeftBound22130.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound105022.bound, LeftBound22130.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound105022.actual selector witness) * (LeftBound22130.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 40) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound110020

namespace LeftBound110025
def owner : Owner := ⟨.program ⟨257⟩, ⟨59517⟩⟩
def transferEvent : Nat := 110025
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 110023 .coefficient, .predecessor 1 110024 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 110023 .coefficient)
      LeftBound110020.bound (LeftBound110020.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events429.exact110022RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound110020.bound, RecordedBoundRefines] <;> decide)
      (LeftBound110020.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 110024 .coefficient)
      LeftBound110015.bound (LeftBound110015.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events429.exact110017RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound110015.bound, RecordedBoundRefines] <;> decide)
      (LeftBound110015.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound110020.bound, LeftBound110015.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound110020.bound, LeftBound110015.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound110020.actual selector witness, LeftBound110015.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound110025

namespace LeftBound110029
def owner : Owner := ⟨.program ⟨257⟩, ⟨59518⟩⟩
def transferEvent : Nat := 110029
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 110027 .coefficient, .predecessor 1 110028 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 110027 .coefficient)
      LeftBound110025.bound (LeftBound110025.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events429.exact110026RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound110025.bound, RecordedBoundRefines] <;> decide)
      (LeftBound110025.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 110028 .coefficient)
      LeftBound22122.bound (LeftBound22122.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events086.exact22123RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound22122.bound, RecordedBoundRefines] <;> decide)
      (LeftBound22122.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound110025.bound, LeftBound22122.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound110025.bound, LeftBound22122.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound110025.actual selector witness, LeftBound22122.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound110029

namespace LeftBound110030
def owner : Owner := ⟨.program ⟨257⟩, ⟨59518⟩⟩
def transferEvent : Nat := 110030
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨117⟩⟩]⟩ [⟨.result 22123 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 22123 .coefficient)
      LeftBound22122.bound (LeftBound22122.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨117⟩⟩) (rawTerms := some (Proof.Events086.exact22123RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound22122.bound, RecordedBoundRefines] <;> decide)
      (LeftBound22122.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound22122.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound22122.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftBound22122.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound110030

namespace LeftBound110035
def owner : Owner := ⟨.program ⟨257⟩, ⟨59519⟩⟩
def transferEvent : Nat := 110035
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 110033 .coefficient) (.predecessor 1 110034 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 110033 .coefficient)
      LeftBound110029.bound (LeftBound110029.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events429.exact110032RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound110029.bound, RecordedBoundRefines] <;> decide)
      (LeftBound110029.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 110034 .coefficient)
      LeftBound22119.bound (LeftBound22119.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events086.exact22120RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound22119.bound, RecordedBoundRefines] <;> decide)
      (LeftBound22119.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound110029.bound LeftBound22119.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound110029.bound, LeftBound22119.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound110029.actual selector witness) * (LeftBound22119.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound110035

namespace LeftBound110036
def owner : Owner := ⟨.program ⟨257⟩, ⟨59519⟩⟩
def transferEvent : Nat := 110036
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨9535⟩⟩]⟩ [⟨.result 22116 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 22116 .coefficient)
      LeftAuthority22115.bound (LeftAuthority22115.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨9535⟩⟩) (rawTerms := some (Proof.Events086.exact22116RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority22115.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority22115.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority22115.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority22115.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority22115.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound110036

namespace LeftBound110037
def owner : Owner := ⟨.program ⟨257⟩, ⟨59519⟩⟩
def transferEvent : Nat := 110037
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 110032 .summary) (.transfer 110036) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 110032 .summary)
      LeftBound110030.bound (LeftBound110030.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨59518⟩⟩) (rawTerms := some (Proof.Events429.exact110032RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound110030.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 110036)
      LeftBound110036.bound (LeftBound110036.actual selector witness) := by
  exact .transfer (LeftBound110036.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound110030.bound LeftBound110036.bound
def bound : CoeffClass := .finite ⟨279172874240, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound110030.bound, LeftBound110036.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound110030.actual selector witness) * (LeftBound110036.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound110037

namespace LeftBound110045
def owner : Owner := ⟨.program ⟨257⟩, ⟨59520⟩⟩
def transferEvent : Nat := 110045
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 110043 .coefficient, .predecessor 1 110044 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 110043 .coefficient)
      LeftBound110035.bound (LeftBound110035.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events429.exact110042RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound110035.bound, RecordedBoundRefines] <;> decide)
      (LeftBound110035.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 110044 .coefficient)
      LeftBound110007.bound (LeftBound110007.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events429.exact110012RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound110007.bound, RecordedBoundRefines] <;> decide)
      (LeftBound110007.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound110035.bound, LeftBound110007.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound110035.bound, LeftBound110007.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound110035.actual selector witness, LeftBound110007.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound110045

namespace LeftBound110047
def owner : Owner := ⟨.program ⟨257⟩, ⟨59520⟩⟩
def transferEvent : Nat := 110047
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 110042 .summary, .result 110012 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 110042 .summary)
      LeftBound110037.bound (LeftBound110037.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨59519⟩⟩) (rawTerms := some (Proof.Events429.exact110042RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound110037.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 110012 .summary)
      LeftBound110009.bound (LeftBound110009.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨59515⟩⟩) (rawTerms := some (Proof.Events429.exact110012RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound110009.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound110037.bound, LeftBound110009.bound]
def bound : CoeffClass := .finite ⟨279188209664, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound110037.bound, LeftBound110009.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound110037.actual selector witness, LeftBound110009.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound110047

namespace LeftBound110051
def owner : Owner := ⟨.program ⟨257⟩, ⟨61471⟩⟩
def transferEvent : Nat := 110051
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 110049 .coefficient) (.predecessor 1 110050 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 110049 .coefficient)
      LeftBound110045.bound (LeftBound110045.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events429.exact110048RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound110045.bound, RecordedBoundRefines] <;> decide)
      (LeftBound110045.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 110050 .coefficient)
      LeftAuthority109983.bound (LeftAuthority109983.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events429.exact109984RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority109983.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority109983.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound110045.bound LeftAuthority109983.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound110045.bound, LeftAuthority109983.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound110045.actual selector witness) * (LeftAuthority109983.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound110051

namespace LeftBound110052
def owner : Owner := ⟨.program ⟨257⟩, ⟨61471⟩⟩
def transferEvent : Nat := 110052
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨61470⟩⟩]⟩ [⟨.result 109984 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 109984 .coefficient)
      LeftAuthority109983.bound (LeftAuthority109983.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨61470⟩⟩) (rawTerms := some (Proof.Events429.exact109984RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority109983.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority109983.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority109983.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority109983.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority109983.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound110052

namespace LeftBound110053
def owner : Owner := ⟨.program ⟨257⟩, ⟨61471⟩⟩
def transferEvent : Nat := 110053
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 110048 .summary) (.transfer 110052) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 110048 .summary)
      LeftBound110047.bound (LeftBound110047.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨59520⟩⟩) (rawTerms := some (Proof.Events429.exact110048RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound110047.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 110052)
      LeftBound110052.bound (LeftBound110052.actual selector witness) := by
  exact .transfer (LeftBound110052.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound110047.bound LeftBound110052.bound
def bound : CoeffClass := .finite ⟨2997760574839177871360, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound110047.bound, LeftBound110052.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound110047.actual selector witness) * (LeftBound110052.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound110053

namespace LeftBound110064
def owner : Owner := ⟨.program ⟨257⟩, ⟨60401⟩⟩
def transferEvent : Nat := 110064
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 110062 .coefficient) (.value (.predecessor 1 110063 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 110062 .coefficient)
      LeftAuthority110060.bound (LeftAuthority110060.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events429.exact110061RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority110060.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority110060.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 110063 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority110060.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority110060.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority110060.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound110064

namespace LeftBound110068
def owner : Owner := ⟨.program ⟨257⟩, ⟨60402⟩⟩
def transferEvent : Nat := 110068
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 110066 .coefficient) (.predecessor 1 110067 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 110066 .coefficient)
      LeftBound105242.bound (LeftBound105242.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events411.exact105245RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound105242.bound, RecordedBoundRefines] <;> decide)
      (LeftBound105242.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 110067 .coefficient)
      LeftBound110064.bound (LeftBound110064.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events429.exact110065RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound110064.bound, RecordedBoundRefines] <;> decide)
      (LeftBound110064.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound105242.bound LeftBound110064.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound105242.bound, LeftBound110064.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound105242.actual selector witness) * (LeftBound110064.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound110068

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
