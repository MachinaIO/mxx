import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard083
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard581
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard594

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound92508
def owner : Owner := ⟨.program ⟨257⟩, ⟨13958⟩⟩
def transferEvent : Nat := 92508
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 92506 .coefficient, .predecessor 1 92507 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 92506 .coefficient)
      LeftBound92503.bound (LeftBound92503.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events361.exact92505RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound92503.bound, RecordedBoundRefines] <;> decide)
      (LeftBound92503.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 92507 .coefficient)
      LeftBound92498.bound (LeftBound92498.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events361.exact92500RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound92498.bound, RecordedBoundRefines] <;> decide)
      (LeftBound92498.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound92503.bound, LeftBound92498.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound92503.bound, LeftBound92498.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound92503.actual selector witness, LeftBound92498.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound92508

namespace LeftBound92512
def owner : Owner := ⟨.program ⟨257⟩, ⟨13959⟩⟩
def transferEvent : Nat := 92512
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 92510 .coefficient, .predecessor 1 92511 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 92510 .coefficient)
      LeftBound92508.bound (LeftBound92508.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events361.exact92509RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound92508.bound, RecordedBoundRefines] <;> decide)
      (LeftBound92508.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 92511 .coefficient)
      LeftBound19116.bound (LeftBound19116.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events074.exact19117RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound19116.bound, RecordedBoundRefines] <;> decide)
      (LeftBound19116.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound92508.bound, LeftBound19116.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound92508.bound, LeftBound19116.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound92508.actual selector witness, LeftBound19116.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound92512

namespace LeftBound92513
def owner : Owner := ⟨.program ⟨257⟩, ⟨13959⟩⟩
def transferEvent : Nat := 92513
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨124⟩⟩]⟩ [⟨.result 19117 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 19117 .coefficient)
      LeftBound19116.bound (LeftBound19116.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨124⟩⟩) (rawTerms := some (Proof.Events074.exact19117RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound19116.bound, RecordedBoundRefines] <;> decide)
      (LeftBound19116.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound19116.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound19116.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftBound19116.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound92513

namespace LeftBound92518
def owner : Owner := ⟨.program ⟨257⟩, ⟨13960⟩⟩
def transferEvent : Nat := 92518
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 92516 .coefficient) (.predecessor 1 92517 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 92516 .coefficient)
      LeftBound92512.bound (LeftBound92512.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events361.exact92515RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound92512.bound, RecordedBoundRefines] <;> decide)
      (LeftBound92512.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 92517 .coefficient)
      LeftBound19113.bound (LeftBound19113.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events074.exact19114RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound19113.bound, RecordedBoundRefines] <;> decide)
      (LeftBound19113.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound92512.bound LeftBound19113.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound92512.bound, LeftBound19113.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound92512.actual selector witness) * (LeftBound19113.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound92518

namespace LeftBound92519
def owner : Owner := ⟨.program ⟨257⟩, ⟨13960⟩⟩
def transferEvent : Nat := 92519
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨9553⟩⟩]⟩ [⟨.result 19110 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 19110 .coefficient)
      LeftAuthority19109.bound (LeftAuthority19109.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨9553⟩⟩) (rawTerms := some (Proof.Events074.exact19110RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority19109.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority19109.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority19109.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority19109.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority19109.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound92519

namespace LeftBound92520
def owner : Owner := ⟨.program ⟨257⟩, ⟨13960⟩⟩
def transferEvent : Nat := 92520
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 92515 .summary) (.transfer 92519) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 92515 .summary)
      LeftBound92513.bound (LeftBound92513.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨13959⟩⟩) (rawTerms := some (Proof.Events361.exact92515RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound92513.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 92519)
      LeftBound92519.bound (LeftBound92519.actual selector witness) := by
  exact .transfer (LeftBound92519.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound92513.bound LeftBound92519.bound
def bound : CoeffClass := .finite ⟨279172874240, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound92513.bound, LeftBound92519.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound92513.actual selector witness) * (LeftBound92519.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound92520

namespace LeftBound92528
def owner : Owner := ⟨.program ⟨257⟩, ⟨37241⟩⟩
def transferEvent : Nat := 92528
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 92526 .coefficient, .predecessor 1 92527 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 92526 .coefficient)
      LeftBound92518.bound (LeftBound92518.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events361.exact92525RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound92518.bound, RecordedBoundRefines] <;> decide)
      (LeftBound92518.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 92527 .coefficient)
      LeftBound92490.bound (LeftBound92490.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events361.exact92495RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound92490.bound, RecordedBoundRefines] <;> decide)
      (LeftBound92490.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound92518.bound, LeftBound92490.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound92518.bound, LeftBound92490.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound92518.actual selector witness, LeftBound92490.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound92528

namespace LeftBound92530
def owner : Owner := ⟨.program ⟨257⟩, ⟨37241⟩⟩
def transferEvent : Nat := 92530
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 92525 .summary, .result 92495 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 92525 .summary)
      LeftBound92520.bound (LeftBound92520.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨13960⟩⟩) (rawTerms := some (Proof.Events361.exact92525RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound92520.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 92495 .summary)
      LeftBound92492.bound (LeftBound92492.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨37240⟩⟩) (rawTerms := some (Proof.Events361.exact92495RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound92492.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound92520.bound, LeftBound92492.bound]
def bound : CoeffClass := .finite ⟨279208656896, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound92520.bound, LeftBound92492.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound92520.actual selector witness, LeftBound92492.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound92530

namespace LeftBound92534
def owner : Owner := ⟨.program ⟨257⟩, ⟨38995⟩⟩
def transferEvent : Nat := 92534
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 92532 .coefficient) (.predecessor 1 92533 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 92532 .coefficient)
      LeftBound92528.bound (LeftBound92528.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events361.exact92531RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound92528.bound, RecordedBoundRefines] <;> decide)
      (LeftBound92528.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 92533 .coefficient)
      LeftAuthority92466.bound (LeftAuthority92466.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events361.exact92467RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority92466.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority92466.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound92528.bound LeftAuthority92466.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound92528.bound, LeftAuthority92466.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound92528.actual selector witness) * (LeftAuthority92466.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound92534

namespace LeftBound92535
def owner : Owner := ⟨.program ⟨257⟩, ⟨38995⟩⟩
def transferEvent : Nat := 92535
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨38994⟩⟩]⟩ [⟨.result 92467 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 92467 .coefficient)
      LeftAuthority92466.bound (LeftAuthority92466.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨38994⟩⟩) (rawTerms := some (Proof.Events361.exact92467RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority92466.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority92466.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority92466.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority92466.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority92466.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound92535

namespace LeftBound92536
def owner : Owner := ⟨.program ⟨257⟩, ⟨38995⟩⟩
def transferEvent : Nat := 92536
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 92531 .summary) (.transfer 92535) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 92531 .summary)
      LeftBound92530.bound (LeftBound92530.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨37241⟩⟩) (rawTerms := some (Proof.Events361.exact92531RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound92530.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 92535)
      LeftBound92535.bound (LeftBound92535.actual selector witness) := by
  exact .transfer (LeftBound92535.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound92530.bound LeftBound92535.bound
def bound : CoeffClass := .finite ⟨2997980125321012183040, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound92530.bound, LeftBound92535.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound92530.actual selector witness) * (LeftBound92535.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound92536

namespace LeftBound92547
def owner : Owner := ⟨.program ⟨257⟩, ⟨37921⟩⟩
def transferEvent : Nat := 92547
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 92545 .coefficient) (.value (.predecessor 1 92546 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 92545 .coefficient)
      LeftAuthority92543.bound (LeftAuthority92543.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events361.exact92544RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority92543.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority92543.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 92546 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority92543.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority92543.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority92543.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound92547

namespace LeftBound92551
def owner : Owner := ⟨.program ⟨257⟩, ⟨37922⟩⟩
def transferEvent : Nat := 92551
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 92549 .coefficient) (.predecessor 1 92550 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 92549 .coefficient)
      LeftBound90617.bound (LeftBound90617.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events353.exact90620RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound90617.bound, RecordedBoundRefines] <;> decide)
      (LeftBound90617.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 92550 .coefficient)
      LeftBound92547.bound (LeftBound92547.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events361.exact92548RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound92547.bound, RecordedBoundRefines] <;> decide)
      (LeftBound92547.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound90617.bound LeftBound92547.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound90617.bound, LeftBound92547.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound90617.actual selector witness) * (LeftBound92547.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound92551

namespace LeftBound92552
def owner : Owner := ⟨.program ⟨257⟩, ⟨37922⟩⟩
def transferEvent : Nat := 92552
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨37919⟩⟩]⟩ [⟨.result 92544 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 92544 .coefficient)
      LeftAuthority92543.bound (LeftAuthority92543.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨37919⟩⟩) (rawTerms := some (Proof.Events361.exact92544RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority92543.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority92543.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority92543.bound []
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority92543.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority92543.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound92552

namespace LeftBound92553
def owner : Owner := ⟨.program ⟨257⟩, ⟨37922⟩⟩
def transferEvent : Nat := 92553
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 90620 .summary) (.transfer 92552) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 90620 .summary)
      LeftBound90618.bound (LeftBound90618.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨9944⟩⟩) (rawTerms := some (Proof.Events353.exact90620RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound90618.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 92552)
      LeftBound92552.bound (LeftBound92552.actual selector witness) := by
  exact .transfer (LeftBound92552.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound90618.bound LeftBound92552.bound
def bound : CoeffClass := .finite ⟨202072841853861888, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound90618.bound, LeftBound92552.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound90618.actual selector witness) * (LeftBound92552.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound92553

namespace LeftBound92632
def owner : Owner := ⟨.program ⟨257⟩, ⟨37235⟩⟩
def transferEvent : Nat := 92632
def frameStart : Nat := 92603
def rule : BoundRule := .product (.predecessor 0 92630 .coefficient) (.predecessor 1 92631 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 92630 .coefficient)
      LeftAuthority92628.bound (LeftAuthority92628.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events361.exact92629RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority92628.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority92628.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 92631 .coefficient)
      LeftAuthority92625.bound (LeftAuthority92625.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events361.exact92626RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority92625.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority92625.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority92628.bound LeftAuthority92625.bound
def bound : CoeffClass := .finite ⟨1764, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority92628.bound, LeftAuthority92625.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority92628.actual selector witness) * (LeftAuthority92625.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound92632

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
