import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard000
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard134
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard2058

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound302515
def owner : Owner := ⟨.program ⟨257⟩, ⟨15240⟩⟩
def transferEvent : Nat := 302515
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 302513 .coefficient) (.predecessor 1 302514 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 302513 .coefficient)
      LeftBound302509.bound (LeftBound302509.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1181.exact302512RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound302509.bound, RecordedBoundRefines] <;> decide)
      (LeftBound302509.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 302514 .coefficient)
      LeftAuthority14683.bound (LeftAuthority14683.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events057.exact14684RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority14683.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority14683.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftBound302509.bound LeftAuthority14683.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound302509.bound, LeftAuthority14683.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftBound302509.actual selector witness) * (LeftAuthority14683.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound302515

namespace LeftBound302516
def owner : Owner := ⟨.program ⟨257⟩, ⟨15240⟩⟩
def transferEvent : Nat := 302516
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨257⟩, ⟨12231⟩⟩], []⟩ [⟨.result 14684 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 14684 .coefficient)
      LeftAuthority14683.bound (LeftAuthority14683.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨12231⟩⟩) (rawTerms := some (Proof.Events057.exact14684RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority14683.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority14683.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority14683.bound []
def bound : CoeffClass := .finite ⟨2, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority14683.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority14683.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound302516

namespace LeftBound302517
def owner : Owner := ⟨.program ⟨257⟩, ⟨15240⟩⟩
def transferEvent : Nat := 302517
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 302512 .summary) (.transfer 302516) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 302512 .summary)
      LeftBound302510.bound (LeftBound302510.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨15239⟩⟩) (rawTerms := some (Proof.Events1181.exact302512RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound302510.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 302516)
      LeftBound302516.bound (LeftBound302516.actual selector witness) := by
  exact .transfer (LeftBound302516.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound302510.bound LeftBound302516.bound
def bound : CoeffClass := .finite ⟨1703936, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound302510.bound, LeftBound302516.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound302510.actual selector witness) * (LeftBound302516.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound302517

namespace LeftBound302523
def owner : Owner := ⟨.program ⟨257⟩, ⟨12232⟩⟩
def transferEvent : Nat := 302523
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 302521 .coefficient) (.predecessor 1 302522 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 302521 .coefficient)
      LeftAuthority14683.bound (LeftAuthority14683.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events057.exact14684RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority14683.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority14683.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 302522 .coefficient)
      LeftBound30.bound (LeftBound30.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events000.exact32RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound30.bound, RecordedBoundRefines] <;> decide)
      (LeftBound30.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32768 ⟨true, false, none, none, none⟩ LeftAuthority14683.bound LeftBound30.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority14683.bound, LeftBound30.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := tensorFactor 32768 ⟨true, false, none, none, none⟩ * (LeftAuthority14683.actual selector witness) * (LeftBound30.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound302523

namespace LeftBound302528
def owner : Owner := ⟨.program ⟨257⟩, ⟨7451⟩⟩
def transferEvent : Nat := 302528
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 302526 .coefficient) (.predecessor 1 302527 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 302526 .coefficient)
      LeftBound26.bound (LeftBound26.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events000.exact27RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound26.bound, RecordedBoundRefines] <;> decide)
      (LeftBound26.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 302527 .coefficient)
      LeftBound25637.bound (LeftBound25637.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events100.exact25638RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound25637.bound, RecordedBoundRefines] <;> decide)
      (LeftBound25637.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound26.bound LeftBound25637.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound26.bound, LeftBound25637.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound26.actual selector witness) * (LeftBound25637.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 40) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound302528

namespace LeftBound302533
def owner : Owner := ⟨.program ⟨257⟩, ⟨12233⟩⟩
def transferEvent : Nat := 302533
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 302531 .coefficient, .predecessor 1 302532 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 302531 .coefficient)
      LeftBound302528.bound (LeftBound302528.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1181.exact302530RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound302528.bound, RecordedBoundRefines] <;> decide)
      (LeftBound302528.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 302532 .coefficient)
      LeftBound302523.bound (LeftBound302523.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1181.exact302525RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound302523.bound, RecordedBoundRefines] <;> decide)
      (LeftBound302523.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound302528.bound, LeftBound302523.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound302528.bound, LeftBound302523.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound302528.actual selector witness, LeftBound302523.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound302533

namespace LeftBound302537
def owner : Owner := ⟨.program ⟨257⟩, ⟨12234⟩⟩
def transferEvent : Nat := 302537
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 302535 .coefficient, .predecessor 1 302536 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 302535 .coefficient)
      LeftBound302533.bound (LeftBound302533.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1181.exact302534RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound302533.bound, RecordedBoundRefines] <;> decide)
      (LeftBound302533.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 302536 .coefficient)
      LeftBound25629.bound (LeftBound25629.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events100.exact25630RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound25629.bound, RecordedBoundRefines] <;> decide)
      (LeftBound25629.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound302533.bound, LeftBound25629.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound302533.bound, LeftBound25629.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound302533.actual selector witness, LeftBound25629.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound302537

namespace LeftBound302538
def owner : Owner := ⟨.program ⟨257⟩, ⟨12234⟩⟩
def transferEvent : Nat := 302538
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨129⟩⟩]⟩ [⟨.result 25630 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 25630 .coefficient)
      LeftBound25629.bound (LeftBound25629.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨129⟩⟩) (rawTerms := some (Proof.Events100.exact25630RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound25629.bound, RecordedBoundRefines] <;> decide)
      (LeftBound25629.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound25629.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound25629.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftBound25629.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound302538

namespace LeftBound302543
def owner : Owner := ⟨.program ⟨257⟩, ⟨12235⟩⟩
def transferEvent : Nat := 302543
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 302541 .coefficient) (.predecessor 1 302542 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 302541 .coefficient)
      LeftBound302537.bound (LeftBound302537.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1181.exact302540RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound302537.bound, RecordedBoundRefines] <;> decide)
      (LeftBound302537.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 302542 .coefficient)
      LeftBound25626.bound (LeftBound25626.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events100.exact25627RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound25626.bound, RecordedBoundRefines] <;> decide)
      (LeftBound25626.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound302537.bound LeftBound25626.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound302537.bound, LeftBound25626.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound302537.actual selector witness) * (LeftBound25626.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound302543

namespace LeftBound302544
def owner : Owner := ⟨.program ⟨257⟩, ⟨12235⟩⟩
def transferEvent : Nat := 302544
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨9568⟩⟩]⟩ [⟨.result 25623 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 25623 .coefficient)
      LeftAuthority25622.bound (LeftAuthority25622.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨9568⟩⟩) (rawTerms := some (Proof.Events100.exact25623RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority25622.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority25622.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority25622.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority25622.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority25622.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound302544

namespace LeftBound302545
def owner : Owner := ⟨.program ⟨257⟩, ⟨12235⟩⟩
def transferEvent : Nat := 302545
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 302540 .summary) (.transfer 302544) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 302540 .summary)
      LeftBound302538.bound (LeftBound302538.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨12234⟩⟩) (rawTerms := some (Proof.Events1181.exact302540RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound302538.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 302544)
      LeftBound302544.bound (LeftBound302544.actual selector witness) := by
  exact .transfer (LeftBound302544.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound302538.bound LeftBound302544.bound
def bound : CoeffClass := .finite ⟨279172874240, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound302538.bound, LeftBound302544.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound302538.actual selector witness) * (LeftBound302544.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound302545

namespace LeftBound302553
def owner : Owner := ⟨.program ⟨257⟩, ⟨15241⟩⟩
def transferEvent : Nat := 302553
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 302551 .coefficient, .predecessor 1 302552 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 302551 .coefficient)
      LeftBound302543.bound (LeftBound302543.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1181.exact302550RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound302543.bound, RecordedBoundRefines] <;> decide)
      (LeftBound302543.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 302552 .coefficient)
      LeftBound302515.bound (LeftBound302515.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1181.exact302520RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound302515.bound, RecordedBoundRefines] <;> decide)
      (LeftBound302515.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound302543.bound, LeftBound302515.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound302543.bound, LeftBound302515.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound302543.actual selector witness, LeftBound302515.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound302553

namespace LeftBound302555
def owner : Owner := ⟨.program ⟨257⟩, ⟨15241⟩⟩
def transferEvent : Nat := 302555
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 302550 .summary, .result 302520 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 302550 .summary)
      LeftBound302545.bound (LeftBound302545.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨12235⟩⟩) (rawTerms := some (Proof.Events1181.exact302550RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound302545.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 302520 .summary)
      LeftBound302517.bound (LeftBound302517.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨15240⟩⟩) (rawTerms := some (Proof.Events1181.exact302520RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound302517.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound302545.bound, LeftBound302517.bound]
def bound : CoeffClass := .finite ⟨279174578176, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound302545.bound, LeftBound302517.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound302545.actual selector witness, LeftBound302517.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound302555

namespace LeftBound302559
def owner : Owner := ⟨.program ⟨257⟩, ⟨17250⟩⟩
def transferEvent : Nat := 302559
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 302557 .coefficient) (.predecessor 1 302558 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 302557 .coefficient)
      LeftBound302553.bound (LeftBound302553.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1181.exact302556RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound302553.bound, RecordedBoundRefines] <;> decide)
      (LeftBound302553.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 302558 .coefficient)
      LeftAuthority302491.bound (LeftAuthority302491.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1181.exact302492RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority302491.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority302491.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound302553.bound LeftAuthority302491.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound302553.bound, LeftAuthority302491.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound302553.actual selector witness) * (LeftAuthority302491.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound302559

namespace LeftBound302560
def owner : Owner := ⟨.program ⟨257⟩, ⟨17250⟩⟩
def transferEvent : Nat := 302560
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨17249⟩⟩]⟩ [⟨.result 302492 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 302492 .coefficient)
      LeftAuthority302491.bound (LeftAuthority302491.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨17249⟩⟩) (rawTerms := some (Proof.Events1181.exact302492RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority302491.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority302491.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority302491.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority302491.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority302491.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound302560

namespace LeftBound302561
def owner : Owner := ⟨.program ⟨257⟩, ⟨17250⟩⟩
def transferEvent : Nat := 302561
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 302556 .summary) (.transfer 302560) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 302556 .summary)
      LeftBound302555.bound (LeftBound302555.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨15241⟩⟩) (rawTerms := some (Proof.Events1181.exact302556RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound302555.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 302560)
      LeftBound302560.bound (LeftBound302560.actual selector witness) := by
  exact .transfer (LeftBound302560.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound302555.bound LeftBound302560.bound
def bound : CoeffClass := .finite ⟨2997614207851288330240, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound302555.bound, LeftBound302560.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound302555.actual selector witness) * (LeftBound302560.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound302561

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
