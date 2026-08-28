import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1088

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound163878
def owner : Owner := ⟨.program ⟨257⟩, ⟨7302⟩⟩
def transferEvent : Nat := 163878
def frameStart : Nat := 163800
def rule : BoundRule := .identity (.predecessor 0 163877 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 163877 .coefficient)
      LeftAuthority163865.bound (LeftAuthority163865.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events640.exact163866RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority163865.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority163865.derived selector witness)

def rawBound : CoeffClass := LeftAuthority163865.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority163865.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority163865.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound163878

namespace LeftBound163882
def owner : Owner := ⟨.program ⟨257⟩, ⟨9567⟩⟩
def transferEvent : Nat := 163882
def frameStart : Nat := 163800
def rule : BoundRule := .product (.predecessor 0 163880 .coefficient) (.predecessor 1 163881 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 163880 .coefficient)
      LeftBound163878.bound (LeftBound163878.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events640.exact163879RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound163878.bound, RecordedBoundRefines] <;> decide)
      (LeftBound163878.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 163881 .coefficient)
      LeftBound163875.bound (LeftBound163875.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events640.exact163876RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound163875.bound, RecordedBoundRefines] <;> decide)
      (LeftBound163875.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound163878.bound LeftBound163875.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound163878.bound, LeftBound163875.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound163878.actual selector witness) * (LeftBound163875.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound163882

namespace LeftBound163887
def owner : Owner := ⟨.program ⟨257⟩, ⟨49445⟩⟩
def transferEvent : Nat := 163887
def frameStart : Nat := 163800
def rule : BoundRule := .sum [.predecessor 0 163885 .coefficient, .predecessor 1 163886 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 163885 .coefficient)
      LeftBound163882.bound (LeftBound163882.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events640.exact163884RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound163882.bound, RecordedBoundRefines] <;> decide)
      (LeftBound163882.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 163886 .coefficient)
      LeftBound163859.bound (LeftBound163859.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events640.exact163861RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound163859.bound, RecordedBoundRefines] <;> decide)
      (LeftBound163859.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound163882.bound, LeftBound163859.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound163882.bound, LeftBound163859.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound163882.actual selector witness, LeftBound163859.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound163887

namespace LeftBound163891
def owner : Owner := ⟨.program ⟨257⟩, ⟨49706⟩⟩
def transferEvent : Nat := 163891
def frameStart : Nat := 163800
def rule : BoundRule := .product (.predecessor 0 163889 .coefficient) (.predecessor 1 163890 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 163889 .coefficient)
      LeftBound163887.bound (LeftBound163887.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events640.exact163888RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound163887.bound, RecordedBoundRefines] <;> decide)
      (LeftBound163887.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 163890 .coefficient)
      LeftAuthority163844.bound (LeftAuthority163844.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events640.exact163845RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority163844.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority163844.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound163887.bound LeftAuthority163844.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound163887.bound, LeftAuthority163844.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound163887.actual selector witness) * (LeftAuthority163844.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound163891

namespace LeftBound163902
def owner : Owner := ⟨.program ⟨257⟩, ⟨48182⟩⟩
def transferEvent : Nat := 163902
def frameStart : Nat := 163800
def rule : BoundRule := .product (.predecessor 0 163900 .coefficient) (.predecessor 1 163901 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 163900 .coefficient)
      LeftAuthority163855.bound (LeftAuthority163855.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events640.exact163856RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority163855.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority163855.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 163901 .coefficient)
      LeftAuthority163898.bound (LeftAuthority163898.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events640.exact163899RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority163898.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority163898.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority163855.bound LeftAuthority163898.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority163855.bound, LeftAuthority163898.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority163855.actual selector witness) * (LeftAuthority163898.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound163902

namespace LeftBound163910
def owner : Owner := ⟨.program ⟨257⟩, ⟨48183⟩⟩
def transferEvent : Nat := 163910
def frameStart : Nat := 163800
def rule : BoundRule := .sum [.predecessor 0 163908 .coefficient, .predecessor 1 163909 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 163908 .coefficient)
      LeftAuthority163906.bound (LeftAuthority163906.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events640.exact163907RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority163906.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority163906.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 163909 .coefficient)
      LeftBound163902.bound (LeftBound163902.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events640.exact163904RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound163902.bound, RecordedBoundRefines] <;> decide)
      (LeftBound163902.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority163906.bound, LeftBound163902.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority163906.bound, LeftBound163902.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority163906.actual selector witness, LeftBound163902.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound163910

namespace LeftBound163914
def owner : Owner := ⟨.program ⟨257⟩, ⟨49707⟩⟩
def transferEvent : Nat := 163914
def frameStart : Nat := 163800
def rule : BoundRule := .sum [.predecessor 0 163912 .coefficient, .predecessor 1 163913 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 163912 .coefficient)
      LeftBound163910.bound (LeftBound163910.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events640.exact163911RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound163910.bound, RecordedBoundRefines] <;> decide)
      (LeftBound163910.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 163913 .coefficient)
      LeftBound163891.bound (LeftBound163891.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events640.exact163896RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound163891.bound, RecordedBoundRefines] <;> decide)
      (LeftBound163891.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound163910.bound, LeftBound163891.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound163910.bound, LeftBound163891.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound163910.actual selector witness, LeftBound163891.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound163914

namespace LeftBound163927
def owner : Owner := ⟨.program ⟨257⟩, ⟨49705⟩⟩
def transferEvent : Nat := 163927
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 163925 .coefficient, .predecessor 1 163926 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 163925 .coefficient)
      LeftBound163748.bound (LeftBound163748.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events640.exact163924RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound163748.bound, RecordedBoundRefines] <;> decide)
      (LeftBound163748.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 163926 .coefficient)
      LeftBound163720.bound (LeftBound163720.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events639.exact163727RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound163720.bound, RecordedBoundRefines] <;> decide)
      (LeftBound163720.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound163748.bound, LeftBound163720.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound163748.bound, LeftBound163720.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound163748.actual selector witness, LeftBound163720.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound163927

namespace LeftBound163930
def owner : Owner := ⟨.program ⟨257⟩, ⟨49705⟩⟩
def transferEvent : Nat := 163930
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 163924 .summary, .result 163727 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 163924 .summary)
      LeftBound163750.bound (LeftBound163750.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨48632⟩⟩) (rawTerms := some (Proof.Events640.exact163924RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound163750.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 163727 .summary)
      LeftBound163722.bound (LeftBound163722.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨49704⟩⟩) (rawTerms := some (Proof.Events639.exact163727RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound163722.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound163750.bound, LeftBound163722.bound]
def bound : CoeffClass := .finite ⟨2998346861024241778688, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound163750.bound, LeftBound163722.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound163750.actual selector witness, LeftBound163722.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound163930

namespace LeftBound163934
def owner : Owner := ⟨.program ⟨257⟩, ⟨50131⟩⟩
def transferEvent : Nat := 163934
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 163932 .coefficient) (.predecessor 1 163933 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 163932 .coefficient)
      LeftBound163927.bound (LeftBound163927.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events640.exact163931RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound163927.bound, RecordedBoundRefines] <;> decide)
      (LeftBound163927.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 163933 .coefficient)
      LeftAuthority163637.bound (LeftAuthority163637.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events639.exact163638RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority163637.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority163637.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound163927.bound LeftAuthority163637.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound163927.bound, LeftAuthority163637.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound163927.actual selector witness) * (LeftAuthority163637.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound163934

namespace LeftBound163935
def owner : Owner := ⟨.program ⟨257⟩, ⟨50131⟩⟩
def transferEvent : Nat := 163935
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨50129⟩⟩]⟩ [⟨.result 163638 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 163638 .coefficient)
      LeftAuthority163637.bound (LeftAuthority163637.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨50129⟩⟩) (rawTerms := some (Proof.Events639.exact163638RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority163637.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority163637.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority163637.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority163637.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority163637.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound163935

namespace LeftBound163936
def owner : Owner := ⟨.program ⟨257⟩, ⟨50131⟩⟩
def transferEvent : Nat := 163936
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 163931 .summary) (.transfer 163935) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 163931 .summary)
      LeftBound163930.bound (LeftBound163930.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨49705⟩⟩) (rawTerms := some (Proof.Events640.exact163931RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound163930.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 163935)
      LeftBound163935.bound (LeftBound163935.actual selector witness) := by
  exact .transfer (LeftBound163935.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound163930.bound LeftBound163935.bound
def bound : CoeffClass := .finite ⟨32194504275408438756654574469120, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound163930.bound, LeftBound163935.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound163930.actual selector witness) * (LeftBound163935.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound163936

namespace LeftBound163947
def owner : Owner := ⟨.program ⟨257⟩, ⟨48978⟩⟩
def transferEvent : Nat := 163947
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 163945 .coefficient) (.value (.predecessor 1 163946 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 163945 .coefficient)
      LeftAuthority163943.bound (LeftAuthority163943.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events640.exact163944RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority163943.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority163943.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 163946 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority163943.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority163943.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority163943.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound163947

namespace LeftBound163951
def owner : Owner := ⟨.program ⟨257⟩, ⟨48979⟩⟩
def transferEvent : Nat := 163951
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 163949 .coefficient) (.predecessor 1 163950 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 163949 .coefficient)
      LeftBound163742.bound (LeftBound163742.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events639.exact163745RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound163742.bound, RecordedBoundRefines] <;> decide)
      (LeftBound163742.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 163950 .coefficient)
      LeftBound163947.bound (LeftBound163947.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events640.exact163948RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound163947.bound, RecordedBoundRefines] <;> decide)
      (LeftBound163947.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound163742.bound LeftBound163947.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound163742.bound, LeftBound163947.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound163742.actual selector witness) * (LeftBound163947.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound163951

namespace LeftBound163952
def owner : Owner := ⟨.program ⟨257⟩, ⟨48979⟩⟩
def transferEvent : Nat := 163952
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨48976⟩⟩]⟩ [⟨.result 163944 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 163944 .coefficient)
      LeftAuthority163943.bound (LeftAuthority163943.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨48976⟩⟩) (rawTerms := some (Proof.Events640.exact163944RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority163943.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority163943.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority163943.bound []
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority163943.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority163943.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound163952

namespace LeftBound163953
def owner : Owner := ⟨.program ⟨257⟩, ⟨48979⟩⟩
def transferEvent : Nat := 163953
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 163745 .summary) (.transfer 163952) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 163745 .summary)
      LeftBound163743.bound (LeftBound163743.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨6466⟩⟩) (rawTerms := some (Proof.Events639.exact163745RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound163743.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 163952)
      LeftBound163952.bound (LeftBound163952.actual selector witness) := by
  exact .transfer (LeftBound163952.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound163743.bound LeftBound163952.bound
def bound : CoeffClass := .finite ⟨202072841853861888, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound163743.bound, LeftBound163952.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound163743.actual selector witness) * (LeftBound163952.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound163953

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
