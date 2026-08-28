import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard051
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1798
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1875

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound277620
def owner : Owner := ⟨.program ⟨257⟩, ⟨29668⟩⟩
def transferEvent : Nat := 277620
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 277618 .coefficient) (.value (.predecessor 1 277619 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 277618 .coefficient)
      LeftAuthority277616.bound (LeftAuthority277616.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1084.exact277617RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority277616.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority277616.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 277619 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority277616.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority277616.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority277616.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound277620

namespace LeftBound277624
def owner : Owner := ⟨.program ⟨257⟩, ⟨29669⟩⟩
def transferEvent : Nat := 277624
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 277622 .coefficient) (.predecessor 1 277623 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 277622 .coefficient)
      LeftBound266117.bound (LeftBound266117.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1039.exact266120RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound266117.bound, RecordedBoundRefines] <;> decide)
      (LeftBound266117.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 277623 .coefficient)
      LeftBound277620.bound (LeftBound277620.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1084.exact277621RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound277620.bound, RecordedBoundRefines] <;> decide)
      (LeftBound277620.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound266117.bound LeftBound277620.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound266117.bound, LeftBound277620.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound266117.actual selector witness) * (LeftBound277620.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound277624

namespace LeftBound277625
def owner : Owner := ⟨.program ⟨257⟩, ⟨29669⟩⟩
def transferEvent : Nat := 277625
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨29666⟩⟩]⟩ [⟨.result 277617 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 277617 .coefficient)
      LeftAuthority277616.bound (LeftAuthority277616.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨29666⟩⟩) (rawTerms := some (Proof.Events1084.exact277617RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority277616.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority277616.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority277616.bound []
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority277616.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority277616.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound277625

namespace LeftBound277626
def owner : Owner := ⟨.program ⟨257⟩, ⟨29669⟩⟩
def transferEvent : Nat := 277626
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 266120 .summary) (.transfer 277625) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 266120 .summary)
      LeftBound266118.bound (LeftBound266118.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨5449⟩⟩) (rawTerms := some (Proof.Events1039.exact266120RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound266118.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 277625)
      LeftBound277625.bound (LeftBound277625.actual selector witness) := by
  exact .transfer (LeftBound277625.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound266118.bound LeftBound277625.bound
def bound : CoeffClass := .finite ⟨202072841853861888, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound266118.bound, LeftBound277625.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound266118.actual selector witness) * (LeftBound277625.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound277626

namespace LeftBound277721
def owner : Owner := ⟨.program ⟨257⟩, ⟨29023⟩⟩
def transferEvent : Nat := 277721
def frameStart : Nat := 277682
def rule : BoundRule := .identity (.predecessor 0 277720 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 277720 .coefficient)
      LeftAuthority277718.bound (LeftAuthority277718.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1084.exact277719RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority277718.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority277718.derived selector witness)

def rawBound : CoeffClass := LeftAuthority277718.bound
def bound : CoeffClass := .finite ⟨36, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority277718.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority277718.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound277721

namespace LeftBound277738
def owner : Owner := ⟨.program ⟨257⟩, ⟨30414⟩⟩
def transferEvent : Nat := 277738
def frameStart : Nat := 277682
def rule : BoundRule := .sum [.predecessor 0 277736 .coefficient, .predecessor 1 277737 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 277736 .coefficient)
      LeftBound277721.bound (LeftBound277721.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound277721.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 277737 .coefficient)
      LeftAuthority277734.bound (LeftAuthority277734.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority277734.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound277721.bound, LeftAuthority277734.bound]
def bound : CoeffClass := .finite ⟨36, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound277721.bound, LeftAuthority277734.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound277721.actual selector witness, LeftAuthority277734.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound277738

namespace LeftBound277741
def owner : Owner := ⟨.program ⟨257⟩, ⟨30415⟩⟩
def transferEvent : Nat := 277741
def frameStart : Nat := 277682
def rule : BoundRule := .identity (.predecessor 0 277740 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 277740 .coefficient)
      LeftBound277738.bound (LeftBound277738.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound277738.derived selector witness)

def rawBound : CoeffClass := LeftBound277738.bound
def bound : CoeffClass := .finite ⟨36, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound277738.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound277738.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound277741

namespace LeftBound277747
def owner : Owner := ⟨.program ⟨257⟩, ⟨30416⟩⟩
def transferEvent : Nat := 277747
def frameStart : Nat := 277682
def rule : BoundRule := .product (.predecessor 0 277745 .coefficient) (.predecessor 1 277746 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 277745 .coefficient)
      LeftAuthority277743.bound (LeftAuthority277743.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1084.exact277744RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority277743.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority277743.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 277746 .coefficient)
      LeftBound277741.bound (LeftBound277741.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1084.exact277742RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound277741.bound, RecordedBoundRefines] <;> decide)
      (LeftBound277741.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority277743.bound LeftBound277741.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority277743.bound, LeftBound277741.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority277743.actual selector witness) * (LeftBound277741.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound277747

namespace LeftBound277755
def owner : Owner := ⟨.program ⟨257⟩, ⟨30417⟩⟩
def transferEvent : Nat := 277755
def frameStart : Nat := 277682
def rule : BoundRule := .sum [.predecessor 0 277753 .coefficient, .predecessor 1 277754 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 277753 .coefficient)
      LeftAuthority277751.bound (LeftAuthority277751.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1084.exact277752RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority277751.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority277751.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 277754 .coefficient)
      LeftBound277747.bound (LeftBound277747.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1084.exact277749RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound277747.bound, RecordedBoundRefines] <;> decide)
      (LeftBound277747.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority277751.bound, LeftBound277747.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority277751.bound, LeftBound277747.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority277751.actual selector witness, LeftBound277747.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound277755

namespace LeftBound277759
def owner : Owner := ⟨.program ⟨257⟩, ⟨30757⟩⟩
def transferEvent : Nat := 277759
def frameStart : Nat := 277682
def rule : BoundRule := .product (.predecessor 0 277757 .coefficient) (.predecessor 1 277758 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 277757 .coefficient)
      LeftBound277755.bound (LeftBound277755.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1084.exact277756RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound277755.bound, RecordedBoundRefines] <;> decide)
      (LeftBound277755.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 277758 .coefficient)
      LeftAuthority277732.bound (LeftAuthority277732.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1084.exact277733RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority277732.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority277732.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound277755.bound LeftAuthority277732.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound277755.bound, LeftAuthority277732.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound277755.actual selector witness) * (LeftAuthority277732.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound277759

namespace LeftBound277770
def owner : Owner := ⟨.program ⟨257⟩, ⟨29197⟩⟩
def transferEvent : Nat := 277770
def frameStart : Nat := 277682
def rule : BoundRule := .product (.predecessor 0 277768 .coefficient) (.predecessor 1 277769 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 277768 .coefficient)
      LeftAuthority277743.bound (LeftAuthority277743.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1084.exact277744RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority277743.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority277743.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 277769 .coefficient)
      LeftAuthority277766.bound (LeftAuthority277766.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1085.exact277767RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority277766.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority277766.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority277743.bound LeftAuthority277766.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority277743.bound, LeftAuthority277766.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority277743.actual selector witness) * (LeftAuthority277766.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound277770

namespace LeftBound277778
def owner : Owner := ⟨.program ⟨257⟩, ⟨29198⟩⟩
def transferEvent : Nat := 277778
def frameStart : Nat := 277682
def rule : BoundRule := .sum [.predecessor 0 277776 .coefficient, .predecessor 1 277777 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 277776 .coefficient)
      LeftAuthority277774.bound (LeftAuthority277774.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1085.exact277775RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority277774.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority277774.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 277777 .coefficient)
      LeftBound277770.bound (LeftBound277770.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1085.exact277772RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound277770.bound, RecordedBoundRefines] <;> decide)
      (LeftBound277770.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority277774.bound, LeftBound277770.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority277774.bound, LeftBound277770.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority277774.actual selector witness, LeftBound277770.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound277778

namespace LeftBound277782
def owner : Owner := ⟨.program ⟨257⟩, ⟨30761⟩⟩
def transferEvent : Nat := 277782
def frameStart : Nat := 277682
def rule : BoundRule := .sum [.predecessor 0 277780 .coefficient, .predecessor 1 277781 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 277780 .coefficient)
      LeftBound277778.bound (LeftBound277778.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1085.exact277779RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound277778.bound, RecordedBoundRefines] <;> decide)
      (LeftBound277778.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 277781 .coefficient)
      LeftBound277759.bound (LeftBound277759.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1085.exact277764RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound277759.bound, RecordedBoundRefines] <;> decide)
      (LeftBound277759.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound277778.bound, LeftBound277759.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound277778.bound, LeftBound277759.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound277778.actual selector witness, LeftBound277759.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound277782

namespace LeftBound277795
def owner : Owner := ⟨.program ⟨257⟩, ⟨30759⟩⟩
def transferEvent : Nat := 277795
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 277793 .coefficient, .predecessor 1 277794 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 277793 .coefficient)
      LeftBound277624.bound (LeftBound277624.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1085.exact277792RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound277624.bound, RecordedBoundRefines] <;> decide)
      (LeftBound277624.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 277794 .coefficient)
      LeftBound277607.bound (LeftBound277607.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1084.exact277614RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound277607.bound, RecordedBoundRefines] <;> decide)
      (LeftBound277607.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound277624.bound, LeftBound277607.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound277624.bound, LeftBound277607.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound277624.actual selector witness, LeftBound277607.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound277795

namespace LeftBound277798
def owner : Owner := ⟨.program ⟨257⟩, ⟨30759⟩⟩
def transferEvent : Nat := 277798
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 277792 .summary, .result 277614 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 277792 .summary)
      LeftBound277626.bound (LeftBound277626.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨29669⟩⟩) (rawTerms := some (Proof.Events1085.exact277792RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound277626.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 277614 .summary)
      LeftBound277609.bound (LeftBound277609.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨30758⟩⟩) (rawTerms := some (Proof.Events1084.exact277614RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound277609.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound277626.bound, LeftBound277609.bound]
def bound : CoeffClass := .finite ⟨32192146870060392302605751287808, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound277626.bound, LeftBound277609.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound277626.actual selector witness, LeftBound277609.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound277798

namespace LeftBound277802
def owner : Owner := ⟨.program ⟨257⟩, ⟨30760⟩⟩
def transferEvent : Nat := 277802
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 277800 .coefficient) (.predecessor 1 277801 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 277800 .coefficient)
      LeftBound277795.bound (LeftBound277795.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1085.exact277799RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound277795.bound, RecordedBoundRefines] <;> decide)
      (LeftBound277795.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 277801 .coefficient)
      LeftBound15661.bound (LeftBound15661.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events061.exact15662RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15661.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15661.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound277795.bound LeftBound15661.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound277795.bound, LeftBound15661.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound277795.actual selector witness) * (LeftBound15661.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound277802

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
