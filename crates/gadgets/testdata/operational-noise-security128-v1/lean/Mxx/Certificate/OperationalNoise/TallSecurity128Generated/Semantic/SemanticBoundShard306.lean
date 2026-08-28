import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard276
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard305

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound50605
def owner : Owner := ⟨.program ⟨257⟩, ⟨67853⟩⟩
def transferEvent : Nat := 50605
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨67850⟩⟩]⟩ [⟨.result 50597 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 50597 .coefficient)
      LeftAuthority50596.bound (LeftAuthority50596.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨67850⟩⟩) (rawTerms := some (Proof.Events197.exact50597RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority50596.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority50596.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority50596.bound []
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority50596.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority50596.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound50605

namespace LeftBound50606
def owner : Owner := ⟨.program ⟨257⟩, ⟨67853⟩⟩
def transferEvent : Nat := 50606
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 46745 .summary) (.transfer 50605) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 46745 .summary)
      LeftBound46743.bound (LeftBound46743.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨11216⟩⟩) (rawTerms := some (Proof.Events182.exact46745RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound46743.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 50605)
      LeftBound50605.bound (LeftBound50605.actual selector witness) := by
  exact .transfer (LeftBound50605.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound46743.bound LeftBound50605.bound
def bound : CoeffClass := .finite ⟨202072841853861888, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound46743.bound, LeftBound50605.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound46743.actual selector witness) * (LeftBound50605.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound50606

namespace LeftBound50685
def owner : Owner := ⟨.program ⟨257⟩, ⟨65662⟩⟩
def transferEvent : Nat := 50685
def frameStart : Nat := 50656
def rule : BoundRule := .product (.predecessor 0 50683 .coefficient) (.predecessor 1 50684 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 50683 .coefficient)
      LeftAuthority50681.bound (LeftAuthority50681.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events197.exact50682RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority50681.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority50681.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 50684 .coefficient)
      LeftAuthority50678.bound (LeftAuthority50678.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events197.exact50679RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority50678.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority50678.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority50681.bound LeftAuthority50678.bound
def bound : CoeffClass := .finite ⟨784, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority50681.bound, LeftAuthority50678.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority50681.actual selector witness) * (LeftAuthority50678.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound50685

namespace LeftBound50689
def owner : Owner := ⟨.program ⟨257⟩, ⟨65663⟩⟩
def transferEvent : Nat := 50689
def frameStart : Nat := 50656
def rule : BoundRule := .identity (.predecessor 0 50688 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 50688 .coefficient)
      LeftBound50685.bound (LeftBound50685.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events197.exact50687RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50685.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50685.derived selector witness)

def rawBound : CoeffClass := LeftBound50685.bound
def bound : CoeffClass := .finite ⟨784, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50685.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound50685.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound50689

namespace LeftBound50706
def owner : Owner := ⟨.program ⟨257⟩, ⟨68959⟩⟩
def transferEvent : Nat := 50706
def frameStart : Nat := 50656
def rule : BoundRule := .sum [.predecessor 0 50704 .coefficient, .predecessor 1 50705 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 50704 .coefficient)
      LeftBound50689.bound (LeftBound50689.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound50689.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 50705 .coefficient)
      LeftAuthority50702.bound (LeftAuthority50702.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority50702.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound50689.bound, LeftAuthority50702.bound]
def bound : CoeffClass := .finite ⟨784, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50689.bound, LeftAuthority50702.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound50689.actual selector witness, LeftAuthority50702.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound50706

namespace LeftBound50709
def owner : Owner := ⟨.program ⟨257⟩, ⟨68960⟩⟩
def transferEvent : Nat := 50709
def frameStart : Nat := 50656
def rule : BoundRule := .identity (.predecessor 0 50708 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 50708 .coefficient)
      LeftBound50706.bound (LeftBound50706.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound50706.derived selector witness)

def rawBound : CoeffClass := LeftBound50706.bound
def bound : CoeffClass := .finite ⟨784, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50706.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound50706.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound50709

namespace LeftBound50715
def owner : Owner := ⟨.program ⟨257⟩, ⟨68961⟩⟩
def transferEvent : Nat := 50715
def frameStart : Nat := 50656
def rule : BoundRule := .product (.predecessor 0 50713 .coefficient) (.predecessor 1 50714 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 50713 .coefficient)
      LeftAuthority50711.bound (LeftAuthority50711.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events198.exact50712RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority50711.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority50711.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 50714 .coefficient)
      LeftBound50709.bound (LeftBound50709.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events198.exact50710RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50709.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50709.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority50711.bound LeftBound50709.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority50711.bound, LeftBound50709.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority50711.actual selector witness) * (LeftBound50709.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound50715

namespace LeftBound50731
def owner : Owner := ⟨.program ⟨257⟩, ⟨9542⟩⟩
def transferEvent : Nat := 50731
def frameStart : Nat := 50656
def rule : BoundRule := .scale (.predecessor 0 50729 .coefficient) (.value (.predecessor 1 50730 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 50729 .coefficient)
      LeftAuthority50727.bound (LeftAuthority50727.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events198.exact50728RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority50727.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority50727.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 50730 .coefficient)
      LeftAuthority50718.bound (LeftAuthority50718.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority50718.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority50727.bound LeftAuthority50718.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority50727.bound, LeftAuthority50718.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority50727.actual selector witness) * (LeftAuthority50718.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound50731

namespace LeftBound50734
def owner : Owner := ⟨.program ⟨257⟩, ⟨7294⟩⟩
def transferEvent : Nat := 50734
def frameStart : Nat := 50656
def rule : BoundRule := .identity (.predecessor 0 50733 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 50733 .coefficient)
      LeftAuthority50721.bound (LeftAuthority50721.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events198.exact50722RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority50721.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority50721.derived selector witness)

def rawBound : CoeffClass := LeftAuthority50721.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority50721.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority50721.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound50734

namespace LeftBound50738
def owner : Owner := ⟨.program ⟨257⟩, ⟨9543⟩⟩
def transferEvent : Nat := 50738
def frameStart : Nat := 50656
def rule : BoundRule := .product (.predecessor 0 50736 .coefficient) (.predecessor 1 50737 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 50736 .coefficient)
      LeftBound50734.bound (LeftBound50734.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events198.exact50735RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50734.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50734.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 50737 .coefficient)
      LeftBound50731.bound (LeftBound50731.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events198.exact50732RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50731.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50731.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound50734.bound LeftBound50731.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50734.bound, LeftBound50731.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound50734.actual selector witness) * (LeftBound50731.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound50738

namespace LeftBound50743
def owner : Owner := ⟨.program ⟨257⟩, ⟨68962⟩⟩
def transferEvent : Nat := 50743
def frameStart : Nat := 50656
def rule : BoundRule := .sum [.predecessor 0 50741 .coefficient, .predecessor 1 50742 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 50741 .coefficient)
      LeftBound50738.bound (LeftBound50738.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events198.exact50740RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50738.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50738.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 50742 .coefficient)
      LeftBound50715.bound (LeftBound50715.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events198.exact50717RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50715.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50715.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound50738.bound, LeftBound50715.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50738.bound, LeftBound50715.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound50738.actual selector witness, LeftBound50715.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound50743

namespace LeftBound50747
def owner : Owner := ⟨.program ⟨257⟩, ⟨69331⟩⟩
def transferEvent : Nat := 50747
def frameStart : Nat := 50656
def rule : BoundRule := .product (.predecessor 0 50745 .coefficient) (.predecessor 1 50746 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 50745 .coefficient)
      LeftBound50743.bound (LeftBound50743.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events198.exact50744RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50743.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50743.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 50746 .coefficient)
      LeftAuthority50700.bound (LeftAuthority50700.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events198.exact50701RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority50700.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority50700.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound50743.bound LeftAuthority50700.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50743.bound, LeftAuthority50700.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound50743.actual selector witness) * (LeftAuthority50700.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound50747

namespace LeftBound50758
def owner : Owner := ⟨.program ⟨257⟩, ⟨65854⟩⟩
def transferEvent : Nat := 50758
def frameStart : Nat := 50656
def rule : BoundRule := .product (.predecessor 0 50756 .coefficient) (.predecessor 1 50757 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 50756 .coefficient)
      LeftAuthority50711.bound (LeftAuthority50711.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events198.exact50712RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority50711.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority50711.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 50757 .coefficient)
      LeftAuthority50754.bound (LeftAuthority50754.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events198.exact50755RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority50754.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority50754.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority50711.bound LeftAuthority50754.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority50711.bound, LeftAuthority50754.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority50711.actual selector witness) * (LeftAuthority50754.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound50758

namespace LeftBound50766
def owner : Owner := ⟨.program ⟨257⟩, ⟨65855⟩⟩
def transferEvent : Nat := 50766
def frameStart : Nat := 50656
def rule : BoundRule := .sum [.predecessor 0 50764 .coefficient, .predecessor 1 50765 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 50764 .coefficient)
      LeftAuthority50762.bound (LeftAuthority50762.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events198.exact50763RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority50762.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority50762.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 50765 .coefficient)
      LeftBound50758.bound (LeftBound50758.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events198.exact50760RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50758.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50758.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority50762.bound, LeftBound50758.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority50762.bound, LeftBound50758.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority50762.actual selector witness, LeftBound50758.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound50766

namespace LeftBound50770
def owner : Owner := ⟨.program ⟨257⟩, ⟨69332⟩⟩
def transferEvent : Nat := 50770
def frameStart : Nat := 50656
def rule : BoundRule := .sum [.predecessor 0 50768 .coefficient, .predecessor 1 50769 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 50768 .coefficient)
      LeftBound50766.bound (LeftBound50766.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events198.exact50767RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50766.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50766.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 50769 .coefficient)
      LeftBound50747.bound (LeftBound50747.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events198.exact50752RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50747.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50747.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound50766.bound, LeftBound50747.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50766.bound, LeftBound50747.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound50766.actual selector witness, LeftBound50747.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound50770

namespace LeftBound50783
def owner : Owner := ⟨.program ⟨257⟩, ⟨69330⟩⟩
def transferEvent : Nat := 50783
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 50781 .coefficient, .predecessor 1 50782 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 50781 .coefficient)
      LeftBound50604.bound (LeftBound50604.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events198.exact50780RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50604.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50604.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 50782 .coefficient)
      LeftBound50587.bound (LeftBound50587.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events197.exact50594RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50587.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50587.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound50604.bound, LeftBound50587.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50604.bound, LeftBound50587.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound50604.actual selector witness, LeftBound50587.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound50783

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
