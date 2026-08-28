import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard885
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard939

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound141729
def owner : Owner := ⟨.program ⟨257⟩, ⟨22302⟩⟩
def transferEvent : Nat := 141729
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨22299⟩⟩]⟩ [⟨.result 141721 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 141721 .coefficient)
      LeftAuthority141720.bound (LeftAuthority141720.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨22299⟩⟩) (rawTerms := some (Proof.Events553.exact141721RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority141720.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority141720.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority141720.bound []
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority141720.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority141720.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound141729

namespace LeftBound141730
def owner : Owner := ⟨.program ⟨257⟩, ⟨22302⟩⟩
def transferEvent : Nat := 141730
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 134495 .summary) (.transfer 141729) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 134495 .summary)
      LeftBound134493.bound (LeftBound134493.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨5473⟩⟩) (rawTerms := some (Proof.Events525.exact134495RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound134493.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 141729)
      LeftBound141729.bound (LeftBound141729.actual selector witness) := by
  exact .transfer (LeftBound141729.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound134493.bound LeftBound141729.bound
def bound : CoeffClass := .finite ⟨202072841853861888, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound134493.bound, LeftBound141729.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound134493.actual selector witness) * (LeftBound141729.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound141730

namespace LeftBound141809
def owner : Owner := ⟨.program ⟨257⟩, ⟨21327⟩⟩
def transferEvent : Nat := 141809
def frameStart : Nat := 141780
def rule : BoundRule := .product (.predecessor 0 141807 .coefficient) (.predecessor 1 141808 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 141807 .coefficient)
      LeftAuthority141805.bound (LeftAuthority141805.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events553.exact141806RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority141805.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority141805.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 141808 .coefficient)
      LeftAuthority141802.bound (LeftAuthority141802.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events553.exact141803RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority141802.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority141802.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority141805.bound LeftAuthority141802.bound
def bound : CoeffClass := .finite ⟨16, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority141805.bound, LeftAuthority141802.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority141805.actual selector witness) * (LeftAuthority141802.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound141809

namespace LeftBound141813
def owner : Owner := ⟨.program ⟨257⟩, ⟨21328⟩⟩
def transferEvent : Nat := 141813
def frameStart : Nat := 141780
def rule : BoundRule := .identity (.predecessor 0 141812 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 141812 .coefficient)
      LeftBound141809.bound (LeftBound141809.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events553.exact141811RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound141809.bound, RecordedBoundRefines] <;> decide)
      (LeftBound141809.derived selector witness)

def rawBound : CoeffClass := LeftBound141809.bound
def bound : CoeffClass := .finite ⟨16, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound141809.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound141809.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound141813

namespace LeftBound141830
def owner : Owner := ⟨.program ⟨257⟩, ⟨23178⟩⟩
def transferEvent : Nat := 141830
def frameStart : Nat := 141780
def rule : BoundRule := .sum [.predecessor 0 141828 .coefficient, .predecessor 1 141829 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 141828 .coefficient)
      LeftBound141813.bound (LeftBound141813.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound141813.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 141829 .coefficient)
      LeftAuthority141826.bound (LeftAuthority141826.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority141826.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound141813.bound, LeftAuthority141826.bound]
def bound : CoeffClass := .finite ⟨16, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound141813.bound, LeftAuthority141826.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound141813.actual selector witness, LeftAuthority141826.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound141830

namespace LeftBound141833
def owner : Owner := ⟨.program ⟨257⟩, ⟨23179⟩⟩
def transferEvent : Nat := 141833
def frameStart : Nat := 141780
def rule : BoundRule := .identity (.predecessor 0 141832 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 141832 .coefficient)
      LeftBound141830.bound (LeftBound141830.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound141830.derived selector witness)

def rawBound : CoeffClass := LeftBound141830.bound
def bound : CoeffClass := .finite ⟨16, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound141830.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound141830.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound141833

namespace LeftBound141839
def owner : Owner := ⟨.program ⟨257⟩, ⟨23180⟩⟩
def transferEvent : Nat := 141839
def frameStart : Nat := 141780
def rule : BoundRule := .product (.predecessor 0 141837 .coefficient) (.predecessor 1 141838 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 141837 .coefficient)
      LeftAuthority141835.bound (LeftAuthority141835.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events554.exact141836RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority141835.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority141835.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 141838 .coefficient)
      LeftBound141833.bound (LeftBound141833.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events554.exact141834RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound141833.bound, RecordedBoundRefines] <;> decide)
      (LeftBound141833.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority141835.bound LeftBound141833.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority141835.bound, LeftBound141833.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority141835.actual selector witness) * (LeftBound141833.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound141839

namespace LeftBound141855
def owner : Owner := ⟨.program ⟨257⟩, ⟨9575⟩⟩
def transferEvent : Nat := 141855
def frameStart : Nat := 141780
def rule : BoundRule := .scale (.predecessor 0 141853 .coefficient) (.value (.predecessor 1 141854 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 141853 .coefficient)
      LeftAuthority141851.bound (LeftAuthority141851.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events554.exact141852RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority141851.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority141851.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 141854 .coefficient)
      LeftAuthority141842.bound (LeftAuthority141842.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority141842.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority141851.bound LeftAuthority141842.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority141851.bound, LeftAuthority141842.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority141851.actual selector witness) * (LeftAuthority141842.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound141855

namespace LeftBound141858
def owner : Owner := ⟨.program ⟨257⟩, ⟨7286⟩⟩
def transferEvent : Nat := 141858
def frameStart : Nat := 141780
def rule : BoundRule := .identity (.predecessor 0 141857 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 141857 .coefficient)
      LeftAuthority141845.bound (LeftAuthority141845.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events554.exact141846RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority141845.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority141845.derived selector witness)

def rawBound : CoeffClass := LeftAuthority141845.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority141845.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority141845.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound141858

namespace LeftBound141862
def owner : Owner := ⟨.program ⟨257⟩, ⟨9576⟩⟩
def transferEvent : Nat := 141862
def frameStart : Nat := 141780
def rule : BoundRule := .product (.predecessor 0 141860 .coefficient) (.predecessor 1 141861 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 141860 .coefficient)
      LeftBound141858.bound (LeftBound141858.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events554.exact141859RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound141858.bound, RecordedBoundRefines] <;> decide)
      (LeftBound141858.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 141861 .coefficient)
      LeftBound141855.bound (LeftBound141855.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events554.exact141856RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound141855.bound, RecordedBoundRefines] <;> decide)
      (LeftBound141855.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound141858.bound LeftBound141855.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound141858.bound, LeftBound141855.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound141858.actual selector witness) * (LeftBound141855.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound141862

namespace LeftBound141867
def owner : Owner := ⟨.program ⟨257⟩, ⟨23181⟩⟩
def transferEvent : Nat := 141867
def frameStart : Nat := 141780
def rule : BoundRule := .sum [.predecessor 0 141865 .coefficient, .predecessor 1 141866 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 141865 .coefficient)
      LeftBound141862.bound (LeftBound141862.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events554.exact141864RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound141862.bound, RecordedBoundRefines] <;> decide)
      (LeftBound141862.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 141866 .coefficient)
      LeftBound141839.bound (LeftBound141839.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events554.exact141841RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound141839.bound, RecordedBoundRefines] <;> decide)
      (LeftBound141839.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound141862.bound, LeftBound141839.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound141862.bound, LeftBound141839.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound141862.actual selector witness, LeftBound141839.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound141867

namespace LeftBound141871
def owner : Owner := ⟨.program ⟨257⟩, ⟨23365⟩⟩
def transferEvent : Nat := 141871
def frameStart : Nat := 141780
def rule : BoundRule := .product (.predecessor 0 141869 .coefficient) (.predecessor 1 141870 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 141869 .coefficient)
      LeftBound141867.bound (LeftBound141867.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events554.exact141868RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound141867.bound, RecordedBoundRefines] <;> decide)
      (LeftBound141867.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 141870 .coefficient)
      LeftAuthority141824.bound (LeftAuthority141824.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events554.exact141825RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority141824.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority141824.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound141867.bound LeftAuthority141824.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound141867.bound, LeftAuthority141824.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound141867.actual selector witness) * (LeftAuthority141824.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound141871

namespace LeftBound141882
def owner : Owner := ⟨.program ⟨257⟩, ⟨21754⟩⟩
def transferEvent : Nat := 141882
def frameStart : Nat := 141780
def rule : BoundRule := .product (.predecessor 0 141880 .coefficient) (.predecessor 1 141881 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 141880 .coefficient)
      LeftAuthority141835.bound (LeftAuthority141835.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events554.exact141836RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority141835.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority141835.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 141881 .coefficient)
      LeftAuthority141878.bound (LeftAuthority141878.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events554.exact141879RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority141878.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority141878.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority141835.bound LeftAuthority141878.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority141835.bound, LeftAuthority141878.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority141835.actual selector witness) * (LeftAuthority141878.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound141882

namespace LeftBound141890
def owner : Owner := ⟨.program ⟨257⟩, ⟨21755⟩⟩
def transferEvent : Nat := 141890
def frameStart : Nat := 141780
def rule : BoundRule := .sum [.predecessor 0 141888 .coefficient, .predecessor 1 141889 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 141888 .coefficient)
      LeftAuthority141886.bound (LeftAuthority141886.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events554.exact141887RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority141886.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority141886.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 141889 .coefficient)
      LeftBound141882.bound (LeftBound141882.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events554.exact141884RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound141882.bound, RecordedBoundRefines] <;> decide)
      (LeftBound141882.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority141886.bound, LeftBound141882.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority141886.bound, LeftBound141882.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority141886.actual selector witness, LeftBound141882.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound141890

namespace LeftBound141894
def owner : Owner := ⟨.program ⟨257⟩, ⟨23366⟩⟩
def transferEvent : Nat := 141894
def frameStart : Nat := 141780
def rule : BoundRule := .sum [.predecessor 0 141892 .coefficient, .predecessor 1 141893 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 141892 .coefficient)
      LeftBound141890.bound (LeftBound141890.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events554.exact141891RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound141890.bound, RecordedBoundRefines] <;> decide)
      (LeftBound141890.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 141893 .coefficient)
      LeftBound141871.bound (LeftBound141871.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events554.exact141876RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound141871.bound, RecordedBoundRefines] <;> decide)
      (LeftBound141871.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound141890.bound, LeftBound141871.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound141890.bound, LeftBound141871.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound141890.actual selector witness, LeftBound141871.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound141894

namespace LeftBound141907
def owner : Owner := ⟨.program ⟨257⟩, ⟨23364⟩⟩
def transferEvent : Nat := 141907
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 141905 .coefficient, .predecessor 1 141906 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 141905 .coefficient)
      LeftBound141728.bound (LeftBound141728.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events554.exact141904RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound141728.bound, RecordedBoundRefines] <;> decide)
      (LeftBound141728.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 141906 .coefficient)
      LeftBound141711.bound (LeftBound141711.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events553.exact141718RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound141711.bound, RecordedBoundRefines] <;> decide)
      (LeftBound141711.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound141728.bound, LeftBound141711.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound141728.bound, LeftBound141711.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound141728.actual selector witness, LeftBound141711.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound141907

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
