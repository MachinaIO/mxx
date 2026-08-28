import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard012
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard056
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard374
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard376
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard474

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound75649
def owner : Owner := ⟨.program ⟨257⟩, ⟨71478⟩⟩
def transferEvent : Nat := 75649
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 75647 .coefficient, .predecessor 1 75648 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 75647 .coefficient)
      LeftBound75585.bound (LeftBound75585.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events295.exact75646RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound75585.bound, RecordedBoundRefines] <;> decide)
      (LeftBound75585.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 75648 .coefficient)
      LeftBound61166.bound (LeftBound61166.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events239.exact61243RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound61166.bound, RecordedBoundRefines] <;> decide)
      (LeftBound61166.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound75585.bound, LeftBound61166.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound75585.bound, LeftBound61166.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound75585.actual selector witness, LeftBound61166.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound75649

namespace LeftBound75669
def owner : Owner := ⟨.program ⟨257⟩, ⟨71478⟩⟩
def transferEvent : Nat := 75669
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 75646 .summary, .result 61243 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 75646 .summary)
      LeftBound75587.bound (LeftBound75587.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨71477⟩⟩) (rawTerms := some (Proof.Events295.exact75646RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound75587.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 61243 .summary)
      LeftBound61204.bound (LeftBound61204.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨67612⟩⟩) (rawTerms := some (Proof.Events239.exact61243RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound61204.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound75587.bound, LeftBound61204.bound]
def bound : CoeffClass := .finite ⟨717315235864259647099013782854474880280923984914290088855535616, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound75587.bound, LeftBound61204.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound75587.actual selector witness, LeftBound61204.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound75669

namespace LeftBound75673
def owner : Owner := ⟨.program ⟨257⟩, ⟨71479⟩⟩
def transferEvent : Nat := 75673
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 75671 .coefficient) (.predecessor 1 75672 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 75671 .coefficient)
      LeftBound75649.bound (LeftBound75649.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events295.exact75670RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound75649.bound, RecordedBoundRefines] <;> decide)
      (LeftBound75649.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 75672 .coefficient)
      LeftBound16093.bound (LeftBound16093.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events062.exact16094RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound16093.bound, RecordedBoundRefines] <;> decide)
      (LeftBound16093.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound75649.bound LeftBound16093.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound75649.bound, LeftBound16093.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound75649.actual selector witness) * (LeftBound16093.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound75673

namespace LeftBound75674
def owner : Owner := ⟨.program ⟨257⟩, ⟨71479⟩⟩
def transferEvent : Nat := 75674
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨7141⟩⟩]⟩ [⟨.result 16090 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 16090 .coefficient)
      LeftAuthority16089.bound (LeftAuthority16089.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨7141⟩⟩) (rawTerms := some (Proof.Events062.exact16090RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority16089.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority16089.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority16089.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority16089.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority16089.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound75674

namespace LeftBound75675
def owner : Owner := ⟨.program ⟨257⟩, ⟨71479⟩⟩
def transferEvent : Nat := 75675
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 75670 .summary) (.transfer 75674) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 75670 .summary)
      LeftBound75669.bound (LeftBound75669.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨71478⟩⟩) (rawTerms := some (Proof.Events295.exact75670RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound75669.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 75674)
      LeftBound75674.bound (LeftBound75674.actual selector witness) := by
  exact .transfer (LeftBound75674.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound75669.bound LeftBound75674.bound
def bound : CoeffClass := .finite ⟨7702113697398803698856913678033037845150209519672183236728648848208035840, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound75669.bound, LeftBound75674.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound75669.actual selector witness) * (LeftBound75674.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound75675

namespace LeftBound75756
def owner : Owner := ⟨.program ⟨257⟩, ⟨10437⟩⟩
def transferEvent : Nat := 75756
def frameStart : Nat := 0
def rule : BoundRule := .identity (.predecessor 0 75751 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 75751 .coefficient)
      LeftAuthority19.bound (LeftAuthority19.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events000.exact20RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority19.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority19.derived selector witness)

def rawBound : CoeffClass := LeftAuthority19.bound
def bound : CoeffClass := .finite ⟨1, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority19.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority19.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound75756

namespace LeftBound75760
def owner : Owner := ⟨.program ⟨257⟩, ⟨10438⟩⟩
def transferEvent : Nat := 75760
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 75758 .coefficient) (.predecessor 1 75759 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 75758 .coefficient)
      LeftBound75756.bound (LeftBound75756.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events295.exact75757RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound75756.bound, RecordedBoundRefines] <;> decide)
      (LeftBound75756.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 75759 .coefficient)
      LeftAuthority1.bound (LeftAuthority1.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events000.exact2RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority1.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority1.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound75756.bound LeftAuthority1.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound75756.bound, LeftAuthority1.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound75756.actual selector witness) * (LeftAuthority1.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 40) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound75760

namespace LeftBound75772
def owner : Owner := ⟨.program ⟨257⟩, ⟨10327⟩⟩
def transferEvent : Nat := 75772
def frameStart : Nat := 0
def rule : BoundRule := .identity (.predecessor 0 75767 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 75767 .coefficient)
      LeftAuthority19.bound (LeftAuthority19.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events000.exact20RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority19.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority19.derived selector witness)

def rawBound : CoeffClass := LeftAuthority19.bound
def bound : CoeffClass := .finite ⟨1, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority19.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority19.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound75772

namespace LeftBound75776
def owner : Owner := ⟨.program ⟨257⟩, ⟨10329⟩⟩
def transferEvent : Nat := 75776
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 75774 .coefficient) (.predecessor 1 75775 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 75774 .coefficient)
      LeftBound75772.bound (LeftBound75772.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events295.exact75773RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound75772.bound, RecordedBoundRefines] <;> decide)
      (LeftBound75772.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 75775 .coefficient)
      LeftAuthority16136.bound (LeftAuthority16136.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events063.exact16137RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority16136.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority16136.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound75772.bound LeftAuthority16136.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound75772.bound, LeftAuthority16136.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound75772.actual selector witness) * (LeftAuthority16136.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 40) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound75776

namespace LeftBound75781
def owner : Owner := ⟨.program ⟨257⟩, ⟨10439⟩⟩
def transferEvent : Nat := 75781
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 75779 .coefficient, .predecessor 1 75780 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 75779 .coefficient)
      LeftBound75776.bound (LeftBound75776.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events296.exact75778RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound75776.bound, RecordedBoundRefines] <;> decide)
      (LeftBound75776.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 75780 .coefficient)
      LeftBound75760.bound (LeftBound75760.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events295.exact75762RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound75760.bound, RecordedBoundRefines] <;> decide)
      (LeftBound75760.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound75776.bound, LeftBound75760.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound75776.bound, LeftBound75760.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound75776.actual selector witness, LeftBound75760.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound75781

namespace LeftBound75785
def owner : Owner := ⟨.program ⟨257⟩, ⟨10440⟩⟩
def transferEvent : Nat := 75785
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 75783 .coefficient, .predecessor 1 75784 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 75783 .coefficient)
      LeftBound75781.bound (LeftBound75781.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events296.exact75782RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound75781.bound, RecordedBoundRefines] <;> decide)
      (LeftBound75781.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 75784 .coefficient)
      LeftAuthority75735.bound (LeftAuthority75735.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events295.exact75736RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority75735.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority75735.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound75781.bound, LeftAuthority75735.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound75781.bound, LeftAuthority75735.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound75781.actual selector witness, LeftAuthority75735.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound75785

namespace LeftBound75786
def owner : Owner := ⟨.program ⟨257⟩, ⟨10440⟩⟩
def transferEvent : Nat := 75786
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨25⟩⟩]⟩ [⟨.result 75736 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 75736 .coefficient)
      LeftAuthority75735.bound (LeftAuthority75735.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨25⟩⟩) (rawTerms := some (Proof.Events295.exact75736RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority75735.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority75735.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority75735.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority75735.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority75735.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound75786

namespace LeftBound75791
def owner : Owner := ⟨.program ⟨257⟩, ⟨67592⟩⟩
def transferEvent : Nat := 75791
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 75789 .coefficient) (.predecessor 1 75790 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 75789 .coefficient)
      LeftBound75785.bound (LeftBound75785.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events296.exact75788RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound75785.bound, RecordedBoundRefines] <;> decide)
      (LeftBound75785.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 75790 .coefficient)
      LeftBound3795.bound (LeftBound3795.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events014.exact3796RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound3795.bound, RecordedBoundRefines] <;> decide)
      (LeftBound3795.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound75785.bound LeftBound3795.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound75785.bound, LeftBound3795.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound75785.actual selector witness) * (LeftBound3795.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound75791

namespace LeftBound75792
def owner : Owner := ⟨.program ⟨257⟩, ⟨67592⟩⟩
def transferEvent : Nat := 75792
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67586⟩⟩], []⟩ [⟨.result 36 .coefficient, true, some 1⟩, ⟨.result 3571 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 36 .coefficient)
      LeftAuthority35.bound (LeftAuthority35.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨6774⟩⟩) (rawTerms := some (Proof.Events000.exact36RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority35.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority35.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 3571 .coefficient)
      LeftAuthority3570.bound (LeftAuthority3570.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨67586⟩⟩) (rawTerms := some (Proof.Events013.exact3571RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority3570.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority3570.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority35.bound [LeftAuthority3570.bound]
def bound : CoeffClass := .finite ⟨4222381728938650955397720, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority35.bound, LeftAuthority3570.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority35.actual selector witness) * ([LeftAuthority3570.actual selector witness].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.cons (.intro (input1 selector witness)) (.nil))
end LeftBound75792

namespace LeftBound75793
def owner : Owner := ⟨.program ⟨257⟩, ⟨67592⟩⟩
def transferEvent : Nat := 75793
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48437⟩⟩], []⟩ [⟨.result 543 .coefficient, true, some 1⟩, ⟨.result 3579 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 543 .coefficient)
      LeftAuthority542.bound (LeftAuthority542.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨6800⟩⟩) (rawTerms := some (Proof.Events002.exact543RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority542.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority542.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 3579 .coefficient)
      LeftAuthority3578.bound (LeftAuthority3578.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨48437⟩⟩) (rawTerms := some (Proof.Events013.exact3579RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority3578.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority3578.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority542.bound [LeftAuthority3578.bound]
def bound : CoeffClass := .finite ⟨230731242018505516688400, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority542.bound, LeftAuthority3578.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority542.actual selector witness) * ([LeftAuthority3578.actual selector witness].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.cons (.intro (input1 selector witness)) (.nil))
end LeftBound75793

namespace LeftBound75794
def owner : Owner := ⟨.program ⟨257⟩, ⟨67592⟩⟩
def transferEvent : Nat := 75794
def frameStart : Nat := 0
def rule : BoundRule := .sum [.transfer 75792, .transfer 75793]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 75792)
      LeftBound75792.bound (LeftBound75792.actual selector witness) := by
  exact .transfer (LeftBound75792.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 75793)
      LeftBound75793.bound (LeftBound75793.actual selector witness) := by
  exact .transfer (LeftBound75793.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound75792.bound, LeftBound75793.bound]
def bound : CoeffClass := .finite ⟨4453112970957156472086120, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound75792.bound, LeftBound75793.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound75792.actual selector witness, LeftBound75793.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound75794

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
