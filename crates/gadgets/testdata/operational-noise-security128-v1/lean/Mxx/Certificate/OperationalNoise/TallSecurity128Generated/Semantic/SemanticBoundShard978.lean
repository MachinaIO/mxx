import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard053
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard054
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard167
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard881
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard883
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard977

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound148462
def owner : Owner := ⟨.program ⟨257⟩, ⟨17181⟩⟩
def transferEvent : Nat := 148462
def frameStart : Nat := 148389
def rule : BoundRule := .sum [.predecessor 0 148460 .coefficient, .predecessor 1 148461 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 148460 .coefficient)
      LeftAuthority148458.bound (LeftAuthority148458.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events579.exact148459RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority148458.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority148458.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 148461 .coefficient)
      LeftBound148454.bound (LeftBound148454.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events579.exact148456RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound148454.bound, RecordedBoundRefines] <;> decide)
      (LeftBound148454.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority148458.bound, LeftBound148454.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority148458.bound, LeftBound148454.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority148458.actual selector witness, LeftBound148454.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound148462

namespace LeftBound148466
def owner : Owner := ⟨.program ⟨257⟩, ⟨17559⟩⟩
def transferEvent : Nat := 148466
def frameStart : Nat := 148389
def rule : BoundRule := .product (.predecessor 0 148464 .coefficient) (.predecessor 1 148465 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 148464 .coefficient)
      LeftBound148462.bound (LeftBound148462.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events579.exact148463RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound148462.bound, RecordedBoundRefines] <;> decide)
      (LeftBound148462.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 148465 .coefficient)
      LeftAuthority148439.bound (LeftAuthority148439.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events579.exact148440RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority148439.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority148439.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound148462.bound LeftAuthority148439.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound148462.bound, LeftAuthority148439.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound148462.actual selector witness) * (LeftAuthority148439.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound148466

namespace LeftBound148477
def owner : Owner := ⟨.program ⟨257⟩, ⟨15921⟩⟩
def transferEvent : Nat := 148477
def frameStart : Nat := 148389
def rule : BoundRule := .product (.predecessor 0 148475 .coefficient) (.predecessor 1 148476 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 148475 .coefficient)
      LeftAuthority148450.bound (LeftAuthority148450.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events579.exact148451RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority148450.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority148450.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 148476 .coefficient)
      LeftAuthority148473.bound (LeftAuthority148473.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events579.exact148474RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority148473.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority148473.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority148450.bound LeftAuthority148473.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority148450.bound, LeftAuthority148473.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority148450.actual selector witness) * (LeftAuthority148473.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound148477

namespace LeftBound148485
def owner : Owner := ⟨.program ⟨257⟩, ⟨15922⟩⟩
def transferEvent : Nat := 148485
def frameStart : Nat := 148389
def rule : BoundRule := .sum [.predecessor 0 148483 .coefficient, .predecessor 1 148484 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 148483 .coefficient)
      LeftAuthority148481.bound (LeftAuthority148481.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events580.exact148482RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority148481.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority148481.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 148484 .coefficient)
      LeftBound148477.bound (LeftBound148477.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events579.exact148479RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound148477.bound, RecordedBoundRefines] <;> decide)
      (LeftBound148477.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority148481.bound, LeftBound148477.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority148481.bound, LeftBound148477.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority148481.actual selector witness, LeftBound148477.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound148485

namespace LeftBound148489
def owner : Owner := ⟨.program ⟨257⟩, ⟨17564⟩⟩
def transferEvent : Nat := 148489
def frameStart : Nat := 148389
def rule : BoundRule := .sum [.predecessor 0 148487 .coefficient, .predecessor 1 148488 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 148487 .coefficient)
      LeftBound148485.bound (LeftBound148485.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events580.exact148486RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound148485.bound, RecordedBoundRefines] <;> decide)
      (LeftBound148485.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 148488 .coefficient)
      LeftBound148466.bound (LeftBound148466.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events579.exact148471RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound148466.bound, RecordedBoundRefines] <;> decide)
      (LeftBound148466.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound148485.bound, LeftBound148466.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound148485.bound, LeftBound148466.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound148485.actual selector witness, LeftBound148466.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound148489

namespace LeftBound148502
def owner : Owner := ⟨.program ⟨257⟩, ⟨17561⟩⟩
def transferEvent : Nat := 148502
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 148500 .coefficient, .predecessor 1 148501 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 148500 .coefficient)
      LeftBound148331.bound (LeftBound148331.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events580.exact148499RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound148331.bound, RecordedBoundRefines] <;> decide)
      (LeftBound148331.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 148501 .coefficient)
      LeftBound148314.bound (LeftBound148314.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events579.exact148321RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound148314.bound, RecordedBoundRefines] <;> decide)
      (LeftBound148314.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound148331.bound, LeftBound148314.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound148331.bound, LeftBound148314.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound148331.actual selector witness, LeftBound148314.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound148502

namespace LeftBound148505
def owner : Owner := ⟨.program ⟨257⟩, ⟨17561⟩⟩
def transferEvent : Nat := 148505
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 148499 .summary, .result 148321 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 148499 .summary)
      LeftBound148333.bound (LeftBound148333.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨16455⟩⟩) (rawTerms := some (Proof.Events580.exact148499RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound148333.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 148321 .summary)
      LeftBound148316.bound (LeftBound148316.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨17560⟩⟩) (rawTerms := some (Proof.Events579.exact148321RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound148316.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound148333.bound, LeftBound148316.bound]
def bound : CoeffClass := .finite ⟨32188807212483706889510625476608, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound148333.bound, LeftBound148316.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound148333.actual selector witness, LeftBound148316.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound148505

namespace LeftBound148509
def owner : Owner := ⟨.program ⟨257⟩, ⟨17562⟩⟩
def transferEvent : Nat := 148509
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 148507 .coefficient) (.predecessor 1 148508 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 148507 .coefficient)
      LeftBound148502.bound (LeftBound148502.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events580.exact148506RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound148502.bound, RecordedBoundRefines] <;> decide)
      (LeftBound148502.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 148508 .coefficient)
      LeftBound15881.bound (LeftBound15881.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events062.exact15882RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15881.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15881.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound148502.bound LeftBound15881.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound148502.bound, LeftBound15881.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound148502.actual selector witness) * (LeftBound15881.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound148509

namespace LeftBound148510
def owner : Owner := ⟨.program ⟨257⟩, ⟨17562⟩⟩
def transferEvent : Nat := 148510
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩ [⟨.result 15878 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 15878 .coefficient)
      LeftAuthority15877.bound (LeftAuthority15877.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨7171⟩⟩) (rawTerms := some (Proof.Events062.exact15878RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority15877.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority15877.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority15877.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority15877.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority15877.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound148510

namespace LeftBound148511
def owner : Owner := ⟨.program ⟨257⟩, ⟨17562⟩⟩
def transferEvent : Nat := 148511
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 148506 .summary) (.transfer 148510) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 148506 .summary)
      LeftBound148505.bound (LeftBound148505.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨17561⟩⟩) (rawTerms := some (Proof.Events580.exact148506RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound148505.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 148510)
      LeftBound148510.bound (LeftBound148510.actual selector witness) := by
  exact .transfer (LeftBound148510.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound148505.bound LeftBound148510.bound
def bound : CoeffClass := .finite ⟨345624685687166110058245054666339432529920, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound148505.bound, LeftBound148510.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound148505.actual selector witness) * (LeftBound148510.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound148511

namespace LeftBound148519
def owner : Owner := ⟨.program ⟨257⟩, ⟨7064⟩⟩
def transferEvent : Nat := 148519
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 148517 .coefficient) (.predecessor 1 148518 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 148517 .coefficient)
      LeftAuthority722.bound (LeftAuthority722.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events002.exact723RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority722.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority722.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 148518 .coefficient)
      LeftBound134401.bound (LeftBound134401.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events525.exact134403RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound134401.bound, RecordedBoundRefines] <;> decide)
      (LeftBound134401.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32768 ⟨true, false, none, none, none⟩ LeftAuthority722.bound LeftBound134401.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority722.bound, LeftBound134401.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := tensorFactor 32768 ⟨true, false, none, none, none⟩ * (LeftAuthority722.actual selector witness) * (LeftBound134401.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound148519

namespace LeftBound148524
def owner : Owner := ⟨.program ⟨257⟩, ⟨7800⟩⟩
def transferEvent : Nat := 148524
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 148522 .coefficient) (.predecessor 1 148523 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 148522 .coefficient)
      LeftBound134272.bound (LeftBound134272.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events524.exact134273RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound134272.bound, RecordedBoundRefines] <;> decide)
      (LeftBound134272.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 148523 .coefficient)
      LeftBound15895.bound (LeftBound15895.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events062.exact15896RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15895.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15895.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound134272.bound LeftBound15895.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound134272.bound, LeftBound15895.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound134272.actual selector witness) * (LeftBound15895.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 40) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound148524

namespace LeftBound148529
def owner : Owner := ⟨.program ⟨257⟩, ⟨9305⟩⟩
def transferEvent : Nat := 148529
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 148527 .coefficient, .predecessor 1 148528 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 148527 .coefficient)
      LeftBound148524.bound (LeftBound148524.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events580.exact148526RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound148524.bound, RecordedBoundRefines] <;> decide)
      (LeftBound148524.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 148528 .coefficient)
      LeftBound148519.bound (LeftBound148519.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events580.exact148521RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound148519.bound, RecordedBoundRefines] <;> decide)
      (LeftBound148519.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound148524.bound, LeftBound148519.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound148524.bound, LeftBound148519.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound148524.actual selector witness, LeftBound148519.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound148529

namespace LeftBound148533
def owner : Owner := ⟨.program ⟨257⟩, ⟨9306⟩⟩
def transferEvent : Nat := 148533
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 148531 .coefficient, .predecessor 1 148532 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 148531 .coefficient)
      LeftBound148529.bound (LeftBound148529.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events580.exact148530RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound148529.bound, RecordedBoundRefines] <;> decide)
      (LeftBound148529.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 148532 .coefficient)
      LeftBound31515.bound (LeftBound31515.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events123.exact31516RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound31515.bound, RecordedBoundRefines] <;> decide)
      (LeftBound31515.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound148529.bound, LeftBound31515.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound148529.bound, LeftBound31515.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound148529.actual selector witness, LeftBound31515.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound148533

namespace LeftBound148534
def owner : Owner := ⟨.program ⟨257⟩, ⟨9306⟩⟩
def transferEvent : Nat := 148534
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨118⟩⟩]⟩ [⟨.result 31516 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 31516 .coefficient)
      LeftBound31515.bound (LeftBound31515.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨118⟩⟩) (rawTerms := some (Proof.Events123.exact31516RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound31515.bound, RecordedBoundRefines] <;> decide)
      (LeftBound31515.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound31515.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound31515.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftBound31515.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound148534

namespace LeftBound148539
def owner : Owner := ⟨.program ⟨257⟩, ⟨9456⟩⟩
def transferEvent : Nat := 148539
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 148537 .coefficient, .predecessor 1 148538 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 148537 .coefficient)
      LeftBound148533.bound (LeftBound148533.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events580.exact148536RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound148533.bound, RecordedBoundRefines] <;> decide)
      (LeftBound148533.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 148538 .coefficient)
      LeftBound148533.bound (LeftBound148533.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events580.exact148536RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound148533.bound, RecordedBoundRefines] <;> decide)
      (LeftBound148533.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound148533.bound, LeftBound148533.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound148533.bound, LeftBound148533.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound148533.actual selector witness, LeftBound148533.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound148539

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
