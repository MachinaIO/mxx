import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard095
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard378
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard402

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound64704
def owner : Owner := ⟨.program ⟨257⟩, ⟨13088⟩⟩
def transferEvent : Nat := 64704
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 64702 .coefficient, .predecessor 1 64703 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 64702 .coefficient)
      LeftBound64699.bound (LeftBound64699.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events252.exact64701RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound64699.bound, RecordedBoundRefines] <;> decide)
      (LeftBound64699.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 64703 .coefficient)
      LeftBound64694.bound (LeftBound64694.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events252.exact64696RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound64694.bound, RecordedBoundRefines] <;> decide)
      (LeftBound64694.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound64699.bound, LeftBound64694.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound64699.bound, LeftBound64694.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound64699.actual selector witness, LeftBound64694.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound64704

namespace LeftBound64708
def owner : Owner := ⟨.program ⟨257⟩, ⟨13089⟩⟩
def transferEvent : Nat := 64708
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 64706 .coefficient, .predecessor 1 64707 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 64706 .coefficient)
      LeftBound64704.bound (LeftBound64704.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events252.exact64705RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound64704.bound, RecordedBoundRefines] <;> decide)
      (LeftBound64704.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 64707 .coefficient)
      LeftBound20619.bound (LeftBound20619.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events080.exact20620RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound20619.bound, RecordedBoundRefines] <;> decide)
      (LeftBound20619.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound64704.bound, LeftBound20619.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound64704.bound, LeftBound20619.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound64704.actual selector witness, LeftBound20619.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound64708

namespace LeftBound64709
def owner : Owner := ⟨.program ⟨257⟩, ⟨13089⟩⟩
def transferEvent : Nat := 64709
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨121⟩⟩]⟩ [⟨.result 20620 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 20620 .coefficient)
      LeftBound20619.bound (LeftBound20619.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨121⟩⟩) (rawTerms := some (Proof.Events080.exact20620RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound20619.bound, RecordedBoundRefines] <;> decide)
      (LeftBound20619.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound20619.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound20619.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftBound20619.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound64709

namespace LeftBound64714
def owner : Owner := ⟨.program ⟨257⟩, ⟨13090⟩⟩
def transferEvent : Nat := 64714
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 64712 .coefficient) (.predecessor 1 64713 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 64712 .coefficient)
      LeftBound64708.bound (LeftBound64708.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events252.exact64711RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound64708.bound, RecordedBoundRefines] <;> decide)
      (LeftBound64708.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 64713 .coefficient)
      LeftBound20616.bound (LeftBound20616.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events080.exact20617RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound20616.bound, RecordedBoundRefines] <;> decide)
      (LeftBound20616.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound64708.bound LeftBound20616.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound64708.bound, LeftBound20616.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound64708.actual selector witness) * (LeftBound20616.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound64714

namespace LeftBound64715
def owner : Owner := ⟨.program ⟨257⟩, ⟨13090⟩⟩
def transferEvent : Nat := 64715
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨9544⟩⟩]⟩ [⟨.result 20613 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 20613 .coefficient)
      LeftAuthority20612.bound (LeftAuthority20612.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨9544⟩⟩) (rawTerms := some (Proof.Events080.exact20613RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority20612.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority20612.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority20612.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority20612.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority20612.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound64715

namespace LeftBound64716
def owner : Owner := ⟨.program ⟨257⟩, ⟨13090⟩⟩
def transferEvent : Nat := 64716
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 64711 .summary) (.transfer 64715) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 64711 .summary)
      LeftBound64709.bound (LeftBound64709.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨13089⟩⟩) (rawTerms := some (Proof.Events252.exact64711RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound64709.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 64715)
      LeftBound64715.bound (LeftBound64715.actual selector witness) := by
  exact .transfer (LeftBound64715.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound64709.bound LeftBound64715.bound
def bound : CoeffClass := .finite ⟨279172874240, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound64709.bound, LeftBound64715.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound64709.actual selector witness) * (LeftBound64715.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound64716

namespace LeftBound64724
def owner : Owner := ⟨.program ⟨257⟩, ⟨26269⟩⟩
def transferEvent : Nat := 64724
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 64722 .coefficient, .predecessor 1 64723 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 64722 .coefficient)
      LeftBound64714.bound (LeftBound64714.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events252.exact64721RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound64714.bound, RecordedBoundRefines] <;> decide)
      (LeftBound64714.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 64723 .coefficient)
      LeftBound64686.bound (LeftBound64686.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events252.exact64691RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound64686.bound, RecordedBoundRefines] <;> decide)
      (LeftBound64686.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound64714.bound, LeftBound64686.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound64714.bound, LeftBound64686.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound64714.actual selector witness, LeftBound64686.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound64724

namespace LeftBound64726
def owner : Owner := ⟨.program ⟨257⟩, ⟨26269⟩⟩
def transferEvent : Nat := 64726
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 64721 .summary, .result 64691 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 64721 .summary)
      LeftBound64716.bound (LeftBound64716.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨13090⟩⟩) (rawTerms := some (Proof.Events252.exact64721RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound64716.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 64691 .summary)
      LeftBound64688.bound (LeftBound64688.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨26268⟩⟩) (rawTerms := some (Proof.Events252.exact64691RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound64688.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound64716.bound, LeftBound64688.bound]
def bound : CoeffClass := .finite ⟨279198433280, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound64716.bound, LeftBound64688.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound64716.actual selector witness, LeftBound64688.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound64726

namespace LeftBound64730
def owner : Owner := ⟨.program ⟨257⟩, ⟨27997⟩⟩
def transferEvent : Nat := 64730
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 64728 .coefficient) (.predecessor 1 64729 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 64728 .coefficient)
      LeftBound64724.bound (LeftBound64724.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events252.exact64727RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound64724.bound, RecordedBoundRefines] <;> decide)
      (LeftBound64724.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 64729 .coefficient)
      LeftAuthority64662.bound (LeftAuthority64662.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events252.exact64663RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority64662.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority64662.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound64724.bound LeftAuthority64662.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound64724.bound, LeftAuthority64662.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound64724.actual selector witness) * (LeftAuthority64662.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound64730

namespace LeftBound64731
def owner : Owner := ⟨.program ⟨257⟩, ⟨27997⟩⟩
def transferEvent : Nat := 64731
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨27996⟩⟩]⟩ [⟨.result 64663 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 64663 .coefficient)
      LeftAuthority64662.bound (LeftAuthority64662.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨27996⟩⟩) (rawTerms := some (Proof.Events252.exact64663RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority64662.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority64662.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority64662.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority64662.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority64662.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound64731

namespace LeftBound64732
def owner : Owner := ⟨.program ⟨257⟩, ⟨27997⟩⟩
def transferEvent : Nat := 64732
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 64727 .summary) (.transfer 64731) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 64727 .summary)
      LeftBound64726.bound (LeftBound64726.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨26269⟩⟩) (rawTerms := some (Proof.Events252.exact64727RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound64726.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 64731)
      LeftBound64731.bound (LeftBound64731.actual selector witness) := by
  exact .transfer (LeftBound64731.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound64726.bound LeftBound64731.bound
def bound : CoeffClass := .finite ⟨2997870350080095027200, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound64726.bound, LeftBound64731.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound64726.actual selector witness) * (LeftBound64731.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound64732

namespace LeftBound64743
def owner : Owner := ⟨.program ⟨257⟩, ⟨26921⟩⟩
def transferEvent : Nat := 64743
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 64741 .coefficient) (.value (.predecessor 1 64742 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 64741 .coefficient)
      LeftAuthority64739.bound (LeftAuthority64739.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events252.exact64740RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority64739.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority64739.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 64742 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority64739.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority64739.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority64739.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound64743

namespace LeftBound64747
def owner : Owner := ⟨.program ⟨257⟩, ⟨26922⟩⟩
def transferEvent : Nat := 64747
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 64745 .coefficient) (.predecessor 1 64746 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 64745 .coefficient)
      LeftBound61367.bound (LeftBound61367.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events239.exact61370RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound61367.bound, RecordedBoundRefines] <;> decide)
      (LeftBound61367.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 64746 .coefficient)
      LeftBound64743.bound (LeftBound64743.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events252.exact64744RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound64743.bound, RecordedBoundRefines] <;> decide)
      (LeftBound64743.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound61367.bound LeftBound64743.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound61367.bound, LeftBound64743.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound61367.actual selector witness) * (LeftBound64743.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound64747

namespace LeftBound64748
def owner : Owner := ⟨.program ⟨257⟩, ⟨26922⟩⟩
def transferEvent : Nat := 64748
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨26919⟩⟩]⟩ [⟨.result 64740 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 64740 .coefficient)
      LeftAuthority64739.bound (LeftAuthority64739.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨26919⟩⟩) (rawTerms := some (Proof.Events252.exact64740RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority64739.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority64739.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority64739.bound []
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority64739.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority64739.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound64748

namespace LeftBound64749
def owner : Owner := ⟨.program ⟨257⟩, ⟨26922⟩⟩
def transferEvent : Nat := 64749
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 61370 .summary) (.transfer 64748) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 61370 .summary)
      LeftBound61368.bound (LeftBound61368.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨10792⟩⟩) (rawTerms := some (Proof.Events239.exact61370RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound61368.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 64748)
      LeftBound64748.bound (LeftBound64748.actual selector witness) := by
  exact .transfer (LeftBound64748.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound61368.bound LeftBound64748.bound
def bound : CoeffClass := .finite ⟨202072841853861888, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound61368.bound, LeftBound64748.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound61368.actual selector witness) * (LeftBound64748.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound64749

namespace LeftBound64828
def owner : Owner := ⟨.program ⟨257⟩, ⟨26263⟩⟩
def transferEvent : Nat := 64828
def frameStart : Nat := 64799
def rule : BoundRule := .product (.predecessor 0 64826 .coefficient) (.predecessor 1 64827 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 64826 .coefficient)
      LeftAuthority64824.bound (LeftAuthority64824.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events253.exact64825RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority64824.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority64824.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 64827 .coefficient)
      LeftAuthority64821.bound (LeftAuthority64821.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events253.exact64822RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority64821.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority64821.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority64824.bound LeftAuthority64821.bound
def bound : CoeffClass := .finite ⟨900, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority64824.bound, LeftAuthority64821.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority64824.actual selector witness) * (LeftAuthority64821.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound64828

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
