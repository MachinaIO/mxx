import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard090
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1489
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1492
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1513

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound224978
def owner : Owner := ⟨.program ⟨257⟩, ⟨36103⟩⟩
def transferEvent : Nat := 224978
def frameStart : Nat := 224919
def rule : BoundRule := .identity (.predecessor 0 224977 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 224977 .coefficient)
      LeftBound224975.bound (LeftBound224975.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound224975.derived selector witness)

def rawBound : CoeffClass := LeftBound224975.bound
def bound : CoeffClass := .finite ⟨40, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound224975.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound224975.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound224978

namespace LeftBound224984
def owner : Owner := ⟨.program ⟨257⟩, ⟨36104⟩⟩
def transferEvent : Nat := 224984
def frameStart : Nat := 224919
def rule : BoundRule := .product (.predecessor 0 224982 .coefficient) (.predecessor 1 224983 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 224982 .coefficient)
      LeftAuthority224980.bound (LeftAuthority224980.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events878.exact224981RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority224980.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority224980.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 224983 .coefficient)
      LeftBound224978.bound (LeftBound224978.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events878.exact224979RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound224978.bound, RecordedBoundRefines] <;> decide)
      (LeftBound224978.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority224980.bound LeftBound224978.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority224980.bound, LeftBound224978.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority224980.actual selector witness) * (LeftBound224978.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound224984

namespace LeftBound224992
def owner : Owner := ⟨.program ⟨257⟩, ⟨36105⟩⟩
def transferEvent : Nat := 224992
def frameStart : Nat := 224919
def rule : BoundRule := .sum [.predecessor 0 224990 .coefficient, .predecessor 1 224991 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 224990 .coefficient)
      LeftAuthority224988.bound (LeftAuthority224988.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events878.exact224989RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority224988.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority224988.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 224991 .coefficient)
      LeftBound224984.bound (LeftBound224984.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events878.exact224986RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound224984.bound, RecordedBoundRefines] <;> decide)
      (LeftBound224984.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority224988.bound, LeftBound224984.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority224988.bound, LeftBound224984.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority224988.actual selector witness, LeftBound224984.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound224992

namespace LeftBound224996
def owner : Owner := ⟨.program ⟨257⟩, ⟨36605⟩⟩
def transferEvent : Nat := 224996
def frameStart : Nat := 224919
def rule : BoundRule := .product (.predecessor 0 224994 .coefficient) (.predecessor 1 224995 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 224994 .coefficient)
      LeftBound224992.bound (LeftBound224992.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events878.exact224993RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound224992.bound, RecordedBoundRefines] <;> decide)
      (LeftBound224992.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 224995 .coefficient)
      LeftAuthority224969.bound (LeftAuthority224969.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events878.exact224970RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority224969.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority224969.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound224992.bound LeftAuthority224969.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound224992.bound, LeftAuthority224969.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound224992.actual selector witness) * (LeftAuthority224969.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound224996

namespace LeftBound225007
def owner : Owner := ⟨.program ⟨257⟩, ⟨34951⟩⟩
def transferEvent : Nat := 225007
def frameStart : Nat := 224919
def rule : BoundRule := .product (.predecessor 0 225005 .coefficient) (.predecessor 1 225006 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 225005 .coefficient)
      LeftAuthority224980.bound (LeftAuthority224980.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events878.exact224981RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority224980.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority224980.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 225006 .coefficient)
      LeftAuthority225003.bound (LeftAuthority225003.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events878.exact225004RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority225003.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority225003.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority224980.bound LeftAuthority225003.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority224980.bound, LeftAuthority225003.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority224980.actual selector witness) * (LeftAuthority225003.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound225007

namespace LeftBound225015
def owner : Owner := ⟨.program ⟨257⟩, ⟨34952⟩⟩
def transferEvent : Nat := 225015
def frameStart : Nat := 224919
def rule : BoundRule := .sum [.predecessor 0 225013 .coefficient, .predecessor 1 225014 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 225013 .coefficient)
      LeftAuthority225011.bound (LeftAuthority225011.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events878.exact225012RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority225011.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority225011.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 225014 .coefficient)
      LeftBound225007.bound (LeftBound225007.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events878.exact225009RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound225007.bound, RecordedBoundRefines] <;> decide)
      (LeftBound225007.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority225011.bound, LeftBound225007.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority225011.bound, LeftBound225007.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority225011.actual selector witness, LeftBound225007.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound225015

namespace LeftBound225019
def owner : Owner := ⟨.program ⟨257⟩, ⟨36608⟩⟩
def transferEvent : Nat := 225019
def frameStart : Nat := 224919
def rule : BoundRule := .sum [.predecessor 0 225017 .coefficient, .predecessor 1 225018 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 225017 .coefficient)
      LeftBound225015.bound (LeftBound225015.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events878.exact225016RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound225015.bound, RecordedBoundRefines] <;> decide)
      (LeftBound225015.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 225018 .coefficient)
      LeftBound224996.bound (LeftBound224996.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events878.exact225001RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound224996.bound, RecordedBoundRefines] <;> decide)
      (LeftBound224996.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound225015.bound, LeftBound224996.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound225015.bound, LeftBound224996.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound225015.actual selector witness, LeftBound224996.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound225019

namespace LeftBound225032
def owner : Owner := ⟨.program ⟨257⟩, ⟨36607⟩⟩
def transferEvent : Nat := 225032
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 225030 .coefficient, .predecessor 1 225031 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 225030 .coefficient)
      LeftBound224861.bound (LeftBound224861.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events879.exact225029RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound224861.bound, RecordedBoundRefines] <;> decide)
      (LeftBound224861.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 225031 .coefficient)
      LeftBound224844.bound (LeftBound224844.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events878.exact224851RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound224844.bound, RecordedBoundRefines] <;> decide)
      (LeftBound224844.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound224861.bound, LeftBound224844.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound224861.bound, LeftBound224844.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound224861.actual selector witness, LeftBound224844.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound225032

namespace LeftBound225035
def owner : Owner := ⟨.program ⟨257⟩, ⟨36607⟩⟩
def transferEvent : Nat := 225035
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 225029 .summary, .result 224851 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 225029 .summary)
      LeftBound224863.bound (LeftBound224863.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨35479⟩⟩) (rawTerms := some (Proof.Events879.exact225029RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound224863.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 224851 .summary)
      LeftBound224846.bound (LeftBound224846.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨36606⟩⟩) (rawTerms := some (Proof.Events878.exact224851RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound224846.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound224863.bound, LeftBound224846.bound]
def bound : CoeffClass := .finite ⟨32192539770951767057087530795008, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound224863.bound, LeftBound224846.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound224863.actual selector witness, LeftBound224846.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound225035

namespace LeftBound225059
def owner : Owner := ⟨.program ⟨257⟩, ⟨28753⟩⟩
def transferEvent : Nat := 225059
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 225057 .coefficient) (.predecessor 1 225058 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 225057 .coefficient)
      LeftAuthority10703.bound (LeftAuthority10703.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events041.exact10704RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority10703.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority10703.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 225058 .coefficient)
      LeftBound222151.bound (LeftBound222151.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events867.exact222153RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound222151.bound, RecordedBoundRefines] <;> decide)
      (LeftBound222151.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32768 ⟨true, false, none, none, none⟩ LeftAuthority10703.bound LeftBound222151.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority10703.bound, LeftBound222151.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := tensorFactor 32768 ⟨true, false, none, none, none⟩ * (LeftAuthority10703.actual selector witness) * (LeftBound222151.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound225059

namespace LeftBound225064
def owner : Owner := ⟨.program ⟨257⟩, ⟨8471⟩⟩
def transferEvent : Nat := 225064
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 225062 .coefficient) (.predecessor 1 225063 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 225062 .coefficient)
      LeftBound222022.bound (LeftBound222022.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events867.exact222023RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound222022.bound, RecordedBoundRefines] <;> decide)
      (LeftBound222022.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 225063 .coefficient)
      LeftBound20085.bound (LeftBound20085.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events078.exact20086RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound20085.bound, RecordedBoundRefines] <;> decide)
      (LeftBound20085.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound222022.bound LeftBound20085.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound222022.bound, LeftBound20085.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound222022.actual selector witness) * (LeftBound20085.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 40) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound225064

namespace LeftBound225069
def owner : Owner := ⟨.program ⟨257⟩, ⟨28754⟩⟩
def transferEvent : Nat := 225069
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 225067 .coefficient, .predecessor 1 225068 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 225067 .coefficient)
      LeftBound225064.bound (LeftBound225064.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events879.exact225066RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound225064.bound, RecordedBoundRefines] <;> decide)
      (LeftBound225064.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 225068 .coefficient)
      LeftBound225059.bound (LeftBound225059.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events879.exact225061RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound225059.bound, RecordedBoundRefines] <;> decide)
      (LeftBound225059.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound225064.bound, LeftBound225059.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound225064.bound, LeftBound225059.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound225064.actual selector witness, LeftBound225059.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound225069

namespace LeftBound225073
def owner : Owner := ⟨.program ⟨257⟩, ⟨28755⟩⟩
def transferEvent : Nat := 225073
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 225071 .coefficient, .predecessor 1 225072 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 225071 .coefficient)
      LeftBound225069.bound (LeftBound225069.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events879.exact225070RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound225069.bound, RecordedBoundRefines] <;> decide)
      (LeftBound225069.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 225072 .coefficient)
      LeftBound20077.bound (LeftBound20077.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events078.exact20078RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound20077.bound, RecordedBoundRefines] <;> decide)
      (LeftBound20077.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound225069.bound, LeftBound20077.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound225069.bound, LeftBound20077.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound225069.actual selector witness, LeftBound20077.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound225073

namespace LeftBound225074
def owner : Owner := ⟨.program ⟨257⟩, ⟨28755⟩⟩
def transferEvent : Nat := 225074
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨105⟩⟩]⟩ [⟨.result 20078 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 20078 .coefficient)
      LeftBound20077.bound (LeftBound20077.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨105⟩⟩) (rawTerms := some (Proof.Events078.exact20078RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound20077.bound, RecordedBoundRefines] <;> decide)
      (LeftBound20077.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound20077.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound20077.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftBound20077.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound225074

namespace LeftBound225079
def owner : Owner := ⟨.program ⟨257⟩, ⟨28756⟩⟩
def transferEvent : Nat := 225079
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 225077 .coefficient) (.predecessor 1 225078 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 225077 .coefficient)
      LeftBound225073.bound (LeftBound225073.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events879.exact225076RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound225073.bound, RecordedBoundRefines] <;> decide)
      (LeftBound225073.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 225078 .coefficient)
      LeftAuthority10706.bound (LeftAuthority10706.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events041.exact10707RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority10706.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority10706.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftBound225073.bound LeftAuthority10706.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound225073.bound, LeftAuthority10706.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftBound225073.actual selector witness) * (LeftAuthority10706.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound225079

namespace LeftBound225080
def owner : Owner := ⟨.program ⟨257⟩, ⟨28756⟩⟩
def transferEvent : Nat := 225080
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨257⟩, ⟨13266⟩⟩], []⟩ [⟨.result 10707 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 10707 .coefficient)
      LeftAuthority10706.bound (LeftAuthority10706.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨13266⟩⟩) (rawTerms := some (Proof.Events041.exact10707RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority10706.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority10706.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority10706.bound []
def bound : CoeffClass := .finite ⟨36, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority10706.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority10706.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound225080

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
