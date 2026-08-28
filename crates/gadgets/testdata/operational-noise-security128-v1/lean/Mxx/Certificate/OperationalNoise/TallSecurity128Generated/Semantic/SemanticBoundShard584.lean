import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard071
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard576
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard579
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard581
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard583

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound91052
def owner : Owner := ⟨.program ⟨257⟩, ⟨14857⟩⟩
def transferEvent : Nat := 91052
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 91050 .coefficient) (.predecessor 1 91051 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 91050 .coefficient)
      LeftAuthority3859.bound (LeftAuthority3859.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events015.exact3860RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority3859.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority3859.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 91051 .coefficient)
      LeftBound90526.bound (LeftBound90526.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events353.exact90528RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound90526.bound, RecordedBoundRefines] <;> decide)
      (LeftBound90526.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32768 ⟨true, false, none, none, none⟩ LeftAuthority3859.bound LeftBound90526.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority3859.bound, LeftBound90526.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := tensorFactor 32768 ⟨true, false, none, none, none⟩ * (LeftAuthority3859.actual selector witness) * (LeftBound90526.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound91052

namespace LeftBound91057
def owner : Owner := ⟨.program ⟨257⟩, ⟨9935⟩⟩
def transferEvent : Nat := 91057
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 91055 .coefficient) (.predecessor 1 91056 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 91055 .coefficient)
      LeftBound90397.bound (LeftBound90397.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events353.exact90398RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound90397.bound, RecordedBoundRefines] <;> decide)
      (LeftBound90397.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 91056 .coefficient)
      LeftBound17621.bound (LeftBound17621.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events068.exact17622RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound17621.bound, RecordedBoundRefines] <;> decide)
      (LeftBound17621.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound90397.bound LeftBound17621.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound90397.bound, LeftBound17621.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound90397.actual selector witness) * (LeftBound17621.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 40) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound91057

namespace LeftBound91062
def owner : Owner := ⟨.program ⟨257⟩, ⟨14858⟩⟩
def transferEvent : Nat := 91062
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 91060 .coefficient, .predecessor 1 91061 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 91060 .coefficient)
      LeftBound91057.bound (LeftBound91057.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events355.exact91059RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound91057.bound, RecordedBoundRefines] <;> decide)
      (LeftBound91057.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 91061 .coefficient)
      LeftBound91052.bound (LeftBound91052.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events355.exact91054RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound91052.bound, RecordedBoundRefines] <;> decide)
      (LeftBound91052.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound91057.bound, LeftBound91052.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound91057.bound, LeftBound91052.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound91057.actual selector witness, LeftBound91052.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound91062

namespace LeftBound91066
def owner : Owner := ⟨.program ⟨257⟩, ⟨14859⟩⟩
def transferEvent : Nat := 91066
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 91064 .coefficient, .predecessor 1 91065 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 91064 .coefficient)
      LeftBound91062.bound (LeftBound91062.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events355.exact91063RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound91062.bound, RecordedBoundRefines] <;> decide)
      (LeftBound91062.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 91065 .coefficient)
      LeftBound17613.bound (LeftBound17613.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events068.exact17614RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound17613.bound, RecordedBoundRefines] <;> decide)
      (LeftBound17613.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound91062.bound, LeftBound17613.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound91062.bound, LeftBound17613.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound91062.actual selector witness, LeftBound17613.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound91066

namespace LeftBound91067
def owner : Owner := ⟨.program ⟨257⟩, ⟨14859⟩⟩
def transferEvent : Nat := 91067
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨127⟩⟩]⟩ [⟨.result 17614 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 17614 .coefficient)
      LeftBound17613.bound (LeftBound17613.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨127⟩⟩) (rawTerms := some (Proof.Events068.exact17614RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound17613.bound, RecordedBoundRefines] <;> decide)
      (LeftBound17613.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound17613.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound17613.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftBound17613.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound91067

namespace LeftBound91072
def owner : Owner := ⟨.program ⟨257⟩, ⟨14860⟩⟩
def transferEvent : Nat := 91072
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 91070 .coefficient) (.predecessor 1 91071 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 91070 .coefficient)
      LeftBound91066.bound (LeftBound91066.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events355.exact91069RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound91066.bound, RecordedBoundRefines] <;> decide)
      (LeftBound91066.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 91071 .coefficient)
      LeftBound17610.bound (LeftBound17610.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events068.exact17611RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound17610.bound, RecordedBoundRefines] <;> decide)
      (LeftBound17610.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound91066.bound LeftBound17610.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound91066.bound, LeftBound17610.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound91066.actual selector witness) * (LeftBound17610.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound91072

namespace LeftBound91073
def owner : Owner := ⟨.program ⟨257⟩, ⟨14860⟩⟩
def transferEvent : Nat := 91073
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨9562⟩⟩]⟩ [⟨.result 17607 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 17607 .coefficient)
      LeftAuthority17606.bound (LeftAuthority17606.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨9562⟩⟩) (rawTerms := some (Proof.Events068.exact17607RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority17606.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority17606.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority17606.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority17606.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority17606.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound91073

namespace LeftBound91074
def owner : Owner := ⟨.program ⟨257⟩, ⟨14860⟩⟩
def transferEvent : Nat := 91074
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 91069 .summary) (.transfer 91073) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 91069 .summary)
      LeftBound91067.bound (LeftBound91067.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨14859⟩⟩) (rawTerms := some (Proof.Events355.exact91069RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound91067.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 91073)
      LeftBound91073.bound (LeftBound91073.actual selector witness) := by
  exact .transfer (LeftBound91073.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound91067.bound LeftBound91073.bound
def bound : CoeffClass := .finite ⟨279172874240, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound91067.bound, LeftBound91073.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound91067.actual selector witness) * (LeftBound91073.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound91074

namespace LeftBound91082
def owner : Owner := ⟨.program ⟨257⟩, ⟨45281⟩⟩
def transferEvent : Nat := 91082
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 91080 .coefficient, .predecessor 1 91081 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 91080 .coefficient)
      LeftBound91072.bound (LeftBound91072.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events355.exact91079RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound91072.bound, RecordedBoundRefines] <;> decide)
      (LeftBound91072.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 91081 .coefficient)
      LeftBound91044.bound (LeftBound91044.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events355.exact91049RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound91044.bound, RecordedBoundRefines] <;> decide)
      (LeftBound91044.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound91072.bound, LeftBound91044.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound91072.bound, LeftBound91044.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound91072.actual selector witness, LeftBound91044.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound91082

namespace LeftBound91084
def owner : Owner := ⟨.program ⟨257⟩, ⟨45281⟩⟩
def transferEvent : Nat := 91084
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 91079 .summary, .result 91049 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 91079 .summary)
      LeftBound91074.bound (LeftBound91074.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨14860⟩⟩) (rawTerms := some (Proof.Events355.exact91079RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound91074.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 91049 .summary)
      LeftBound91046.bound (LeftBound91046.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨45280⟩⟩) (rawTerms := some (Proof.Events355.exact91049RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound91046.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound91074.bound, LeftBound91046.bound]
def bound : CoeffClass := .finite ⟨279222288384, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound91074.bound, LeftBound91046.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound91074.actual selector witness, LeftBound91046.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound91084

namespace LeftBound91088
def owner : Owner := ⟨.program ⟨257⟩, ⟨47035⟩⟩
def transferEvent : Nat := 91088
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 91086 .coefficient) (.predecessor 1 91087 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 91086 .coefficient)
      LeftBound91082.bound (LeftBound91082.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events355.exact91085RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound91082.bound, RecordedBoundRefines] <;> decide)
      (LeftBound91082.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 91087 .coefficient)
      LeftAuthority91020.bound (LeftAuthority91020.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events355.exact91021RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority91020.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority91020.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound91082.bound LeftAuthority91020.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound91082.bound, LeftAuthority91020.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound91082.actual selector witness) * (LeftAuthority91020.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound91088

namespace LeftBound91089
def owner : Owner := ⟨.program ⟨257⟩, ⟨47035⟩⟩
def transferEvent : Nat := 91089
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨47034⟩⟩]⟩ [⟨.result 91021 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 91021 .coefficient)
      LeftAuthority91020.bound (LeftAuthority91020.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨47034⟩⟩) (rawTerms := some (Proof.Events355.exact91021RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority91020.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority91020.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority91020.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority91020.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority91020.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound91089

namespace LeftBound91090
def owner : Owner := ⟨.program ⟨257⟩, ⟨47035⟩⟩
def transferEvent : Nat := 91090
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 91085 .summary) (.transfer 91089) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 91085 .summary)
      LeftBound91084.bound (LeftBound91084.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨45281⟩⟩) (rawTerms := some (Proof.Events355.exact91085RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound91084.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 91089)
      LeftBound91089.bound (LeftBound91089.actual selector witness) := by
  exact .transfer (LeftBound91089.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound91084.bound LeftBound91089.bound
def bound : CoeffClass := .finite ⟨2998126492308901724160, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound91084.bound, LeftBound91089.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound91084.actual selector witness) * (LeftBound91089.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound91090

namespace LeftBound91101
def owner : Owner := ⟨.program ⟨257⟩, ⟨45961⟩⟩
def transferEvent : Nat := 91101
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 91099 .coefficient) (.value (.predecessor 1 91100 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 91099 .coefficient)
      LeftAuthority91097.bound (LeftAuthority91097.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events355.exact91098RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority91097.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority91097.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 91100 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority91097.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority91097.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority91097.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound91101

namespace LeftBound91105
def owner : Owner := ⟨.program ⟨257⟩, ⟨45962⟩⟩
def transferEvent : Nat := 91105
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 91103 .coefficient) (.predecessor 1 91104 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 91103 .coefficient)
      LeftBound90617.bound (LeftBound90617.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events353.exact90620RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound90617.bound, RecordedBoundRefines] <;> decide)
      (LeftBound90617.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 91104 .coefficient)
      LeftBound91101.bound (LeftBound91101.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events355.exact91102RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound91101.bound, RecordedBoundRefines] <;> decide)
      (LeftBound91101.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound90617.bound LeftBound91101.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound90617.bound, LeftBound91101.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound90617.actual selector witness) * (LeftBound91101.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound91105

namespace LeftBound91106
def owner : Owner := ⟨.program ⟨257⟩, ⟨45962⟩⟩
def transferEvent : Nat := 91106
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨45959⟩⟩]⟩ [⟨.result 91098 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 91098 .coefficient)
      LeftAuthority91097.bound (LeftAuthority91097.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨45959⟩⟩) (rawTerms := some (Proof.Events355.exact91098RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority91097.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority91097.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority91097.bound []
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority91097.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority91097.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound91106

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
