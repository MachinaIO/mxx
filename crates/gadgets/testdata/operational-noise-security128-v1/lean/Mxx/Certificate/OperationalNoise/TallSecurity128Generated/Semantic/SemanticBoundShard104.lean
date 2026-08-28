import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard068
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard103

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound21666
def owner : Owner := ⟨.program ⟨257⟩, ⟨64344⟩⟩
def transferEvent : Nat := 21666
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 21661 .summary) (.transfer 21665) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 21661 .summary)
      LeftBound21660.bound (LeftBound21660.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨62239⟩⟩) (rawTerms := some (Proof.Events084.exact21661RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound21660.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 21665)
      LeftBound21665.bound (LeftBound21665.actual selector witness) := by
  exact .transfer (LeftBound21665.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound21660.bound LeftBound21665.bound
def bound : CoeffClass := .finite ⟨2997797166586150256640, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21660.bound, LeftBound21665.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound21660.actual selector witness) * (LeftBound21665.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound21666

namespace LeftBound21677
def owner : Owner := ⟨.program ⟨257⟩, ⟨63284⟩⟩
def transferEvent : Nat := 21677
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 21675 .coefficient) (.value (.predecessor 1 21676 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 21675 .coefficient)
      LeftAuthority21673.bound (LeftAuthority21673.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events084.exact21674RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority21673.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority21673.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 21676 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority21673.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority21673.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority21673.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound21677

namespace LeftBound21681
def owner : Owner := ⟨.program ⟨257⟩, ⟨63285⟩⟩
def transferEvent : Nat := 21681
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 21679 .coefficient) (.predecessor 1 21680 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 21679 .coefficient)
      LeftBound17166.bound (LeftBound17166.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events067.exact17169RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound17166.bound, RecordedBoundRefines] <;> decide)
      (LeftBound17166.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 21680 .coefficient)
      LeftBound21677.bound (LeftBound21677.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events084.exact21678RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21677.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21677.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound17166.bound LeftBound21677.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound17166.bound, LeftBound21677.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound17166.actual selector witness) * (LeftBound21677.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound21681

namespace LeftBound21682
def owner : Owner := ⟨.program ⟨257⟩, ⟨63285⟩⟩
def transferEvent : Nat := 21682
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨63282⟩⟩]⟩ [⟨.result 21674 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 21674 .coefficient)
      LeftAuthority21673.bound (LeftAuthority21673.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨63282⟩⟩) (rawTerms := some (Proof.Events084.exact21674RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority21673.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority21673.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority21673.bound []
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority21673.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority21673.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound21682

namespace LeftBound21683
def owner : Owner := ⟨.program ⟨257⟩, ⟨63285⟩⟩
def transferEvent : Nat := 21683
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 17169 .summary) (.transfer 21682) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 17169 .summary)
      LeftBound17167.bound (LeftBound17167.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨5443⟩⟩) (rawTerms := some (Proof.Events067.exact17169RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound17167.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 21682)
      LeftBound21682.bound (LeftBound21682.actual selector witness) := by
  exact .transfer (LeftBound21682.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound17167.bound LeftBound21682.bound
def bound : CoeffClass := .finite ⟨202072841853861888, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound17167.bound, LeftBound21682.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound17167.actual selector witness) * (LeftBound21682.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound21683

namespace LeftBound21762
def owner : Owner := ⟨.program ⟨257⟩, ⟨62232⟩⟩
def transferEvent : Nat := 21762
def frameStart : Nat := 21733
def rule : BoundRule := .product (.predecessor 0 21760 .coefficient) (.predecessor 1 21761 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 21760 .coefficient)
      LeftAuthority21758.bound (LeftAuthority21758.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events084.exact21759RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority21758.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority21758.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 21761 .coefficient)
      LeftAuthority21755.bound (LeftAuthority21755.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events084.exact21756RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority21755.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority21755.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority21758.bound LeftAuthority21755.bound
def bound : CoeffClass := .finite ⟨484, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority21758.bound, LeftAuthority21755.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority21758.actual selector witness) * (LeftAuthority21755.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound21762

namespace LeftBound21766
def owner : Owner := ⟨.program ⟨257⟩, ⟨62233⟩⟩
def transferEvent : Nat := 21766
def frameStart : Nat := 21733
def rule : BoundRule := .identity (.predecessor 0 21765 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 21765 .coefficient)
      LeftBound21762.bound (LeftBound21762.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events085.exact21764RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21762.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21762.derived selector witness)

def rawBound : CoeffClass := LeftBound21762.bound
def bound : CoeffClass := .finite ⟨484, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21762.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound21762.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound21766

namespace LeftBound21783
def owner : Owner := ⟨.program ⟨257⟩, ⟨64170⟩⟩
def transferEvent : Nat := 21783
def frameStart : Nat := 21733
def rule : BoundRule := .sum [.predecessor 0 21781 .coefficient, .predecessor 1 21782 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 21781 .coefficient)
      LeftBound21766.bound (LeftBound21766.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound21766.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 21782 .coefficient)
      LeftAuthority21779.bound (LeftAuthority21779.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority21779.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound21766.bound, LeftAuthority21779.bound]
def bound : CoeffClass := .finite ⟨484, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21766.bound, LeftAuthority21779.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound21766.actual selector witness, LeftAuthority21779.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound21783

namespace LeftBound21786
def owner : Owner := ⟨.program ⟨257⟩, ⟨64171⟩⟩
def transferEvent : Nat := 21786
def frameStart : Nat := 21733
def rule : BoundRule := .identity (.predecessor 0 21785 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 21785 .coefficient)
      LeftBound21783.bound (LeftBound21783.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound21783.derived selector witness)

def rawBound : CoeffClass := LeftBound21783.bound
def bound : CoeffClass := .finite ⟨484, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21783.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound21783.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound21786

namespace LeftBound21792
def owner : Owner := ⟨.program ⟨257⟩, ⟨64172⟩⟩
def transferEvent : Nat := 21792
def frameStart : Nat := 21733
def rule : BoundRule := .product (.predecessor 0 21790 .coefficient) (.predecessor 1 21791 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 21790 .coefficient)
      LeftAuthority21788.bound (LeftAuthority21788.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events085.exact21789RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority21788.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority21788.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 21791 .coefficient)
      LeftBound21786.bound (LeftBound21786.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events085.exact21787RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21786.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21786.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority21788.bound LeftBound21786.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority21788.bound, LeftBound21786.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority21788.actual selector witness) * (LeftBound21786.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound21792

namespace LeftBound21808
def owner : Owner := ⟨.program ⟨257⟩, ⟨9539⟩⟩
def transferEvent : Nat := 21808
def frameStart : Nat := 21733
def rule : BoundRule := .scale (.predecessor 0 21806 .coefficient) (.value (.predecessor 1 21807 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 21806 .coefficient)
      LeftAuthority21804.bound (LeftAuthority21804.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events085.exact21805RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority21804.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority21804.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 21807 .coefficient)
      LeftAuthority21795.bound (LeftAuthority21795.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority21795.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority21804.bound LeftAuthority21795.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority21804.bound, LeftAuthority21795.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority21804.actual selector witness) * (LeftAuthority21795.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound21808

namespace LeftBound21811
def owner : Owner := ⟨.program ⟨257⟩, ⟨7293⟩⟩
def transferEvent : Nat := 21811
def frameStart : Nat := 21733
def rule : BoundRule := .identity (.predecessor 0 21810 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 21810 .coefficient)
      LeftAuthority21798.bound (LeftAuthority21798.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events085.exact21799RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority21798.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority21798.derived selector witness)

def rawBound : CoeffClass := LeftAuthority21798.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority21798.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority21798.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound21811

namespace LeftBound21815
def owner : Owner := ⟨.program ⟨257⟩, ⟨9540⟩⟩
def transferEvent : Nat := 21815
def frameStart : Nat := 21733
def rule : BoundRule := .product (.predecessor 0 21813 .coefficient) (.predecessor 1 21814 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 21813 .coefficient)
      LeftBound21811.bound (LeftBound21811.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events085.exact21812RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21811.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21811.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 21814 .coefficient)
      LeftBound21808.bound (LeftBound21808.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events085.exact21809RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21808.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21808.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound21811.bound LeftBound21808.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21811.bound, LeftBound21808.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound21811.actual selector witness) * (LeftBound21808.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound21815

namespace LeftBound21820
def owner : Owner := ⟨.program ⟨257⟩, ⟨64173⟩⟩
def transferEvent : Nat := 21820
def frameStart : Nat := 21733
def rule : BoundRule := .sum [.predecessor 0 21818 .coefficient, .predecessor 1 21819 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 21818 .coefficient)
      LeftBound21815.bound (LeftBound21815.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events085.exact21817RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21815.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21815.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 21819 .coefficient)
      LeftBound21792.bound (LeftBound21792.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events085.exact21794RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21792.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21792.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound21815.bound, LeftBound21792.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21815.bound, LeftBound21792.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound21815.actual selector witness, LeftBound21792.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound21820

namespace LeftBound21824
def owner : Owner := ⟨.program ⟨257⟩, ⟨64346⟩⟩
def transferEvent : Nat := 21824
def frameStart : Nat := 21733
def rule : BoundRule := .product (.predecessor 0 21822 .coefficient) (.predecessor 1 21823 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 21822 .coefficient)
      LeftBound21820.bound (LeftBound21820.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events085.exact21821RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21820.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21820.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 21823 .coefficient)
      LeftAuthority21777.bound (LeftAuthority21777.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events085.exact21778RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority21777.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority21777.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound21820.bound LeftAuthority21777.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21820.bound, LeftAuthority21777.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound21820.actual selector witness) * (LeftAuthority21777.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound21824

namespace LeftBound21835
def owner : Owner := ⟨.program ⟨257⟩, ⟨62740⟩⟩
def transferEvent : Nat := 21835
def frameStart : Nat := 21733
def rule : BoundRule := .product (.predecessor 0 21833 .coefficient) (.predecessor 1 21834 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 21833 .coefficient)
      LeftAuthority21788.bound (LeftAuthority21788.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events085.exact21789RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority21788.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority21788.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 21834 .coefficient)
      LeftAuthority21831.bound (LeftAuthority21831.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events085.exact21832RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority21831.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority21831.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority21788.bound LeftAuthority21831.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority21788.bound, LeftAuthority21831.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority21788.actual selector witness) * (LeftAuthority21831.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound21835

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
